from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict, Counter
import pickle as pkl
import pandas as pd
import numpy as np
import json, re, os, spacy, nltk

import lemminflect
import random
import string
from nltk.corpus import wordnet as wn
from collections import Counter
from nltk.metrics.distance import edit_distance
from nltk.stem.wordnet import WordNetLemmatizer
import neuralcoref


class BaseDialect(object):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        self.string = ""
        self.rules = {}
        self.doc = None
        self.tokens = []
        self.end_idx = 0
        self.morphosyntax = morphosyntax
        self.modification_counter = Counter()
        self.lexical_swaps = lexical_swaps

        neuralcoref.add_to_pipe(spacy.load("en_core_web_sm"))
        self.nlp = spacy.load("en_core_web_sm")

        self.OBJECTS = {"dobj", "iobj", "obj", "pobj", "obl", "attr"}

        self.NEGATIVES = {
            "anybody": "nobody",
            "Anybody": "Nobody",
            "anyone": "no one",
            "Anyone": "No one",
            "anywhere": "nowhere",
            "Anywhere": "Nowhere",
            "anything": "nothing",
            "Anything": "Nothing",
            "ever": "never",
            "Ever": "Never",
            "sometimes": "never",
            "Sometimes": "Never",
            "something": "nothing",
            "Something": "Nothing",
            "someone": "no one",
            "Someone": "No one",
            "one": "none",
            "One": "None",
        }

        self.MODALS = {
            "should",
            "ought to",
            "can",
            "am able to",
            "is able to",
            "could",
            "was able to",
            "were able to",
            "must",
            "has to",
            "will",
            "shall",
            "may",
            "might",
            "would",
        }
        
        self.PAST_MODAL_MAPPING = {
            "would": "will",
            "might": "can",
            "could": "can"
        }

        self.POSSESSIVES = {
            "you": "your",
            "You": "Your",
            "yourself": "your",
            "Yourself": "Your",
            "him": "his",
            "Him": "His",
            "her": "her",
            "Her": "Her",
            "them": "they",
            "Them": "They",
        }

        self.PLURAL_DETERMINERS = {
            "this": "these",
            "that": "those",
            "much": "many",
            "This": "These",
            "That": "Those",
            "Much": "Many",
        }
        
        self.PRONOUN_OBJ_TO_SUBJ = {
            "him": "he",
            "her": "she",
            "them": "they",
            "us": "we",
            "me": "I",
            "you": "you"
        }

    def __getstate__(self):
        return {"dialect": self.dialect_name}

    def __setstate__(self):
        return self

    def __hash__(self):
        return hash(self.dialect_name)

    def __str__(self):
        return self.dialect_name

    def surface_sub(self, string):
        """Cleaning up the mess left by the morphosyntactic transformations (analogous to surface-structure in Chomsky's surface/deep structure distinction)"""
        for method in self.surface_subs:
            string = method(string)
        return string.strip()

    def convert_sae_to_dialect(self, string):
        """Full Conversion Pipeline"""
        self.update(string)
        for method in self.morphosyntax_transforms:
            method()

        transformed = self.surface_sub(self.compile_from_rules())
        return (
            self.capitalize(transformed) if self.is_capitalized(string) else transformed
        )

    def clear(self):
        """clear all memory of previous string"""
        self.string = ""
        self.rules = {}
        self.doc = None
        self.tokens = []
        self.end_idx = 0
        self.modification_counter = Counter()

    def update(self, string):
        """update memory to the current string"""
        self.clear()
        self.string = string
        self.doc = self.nlp(string)
        self.tokens = list(self.doc)
        self.end_idx = len(string) - 1

    def set_rule(self, token, value, origin=None, check_capital=True):
        """rewrites @token with the value @value"""

        if check_capital and token.text[0].isupper() and len(value):
            remainder = ""
            if len(value)>1:
                remainder = value[1:]
            value = value[0].capitalize() + remainder

        self.rules[
            (
                token.idx,  # starting with this token
                token.idx + len(token)
                if token.idx + len(token) < len(self.string)
                else self.end_idx,  # and ending right before the next
            )
        ] = {"value": value, "type": origin}

        if origin:
            self.modification_counter[origin] += 1
            self.modification_counter["total"] += 1

    def get_full_word(self, token):
        """spaCy tokenizes [don't] into [do, n't], and this method can be used to recover the full word don't given either do or n't"""
        start = 0
        end = self.end_idx
        for idx in range(token.idx, -1, -1):
            if self.string[idx] not in string.ascii_letters + "'’-":
                start = idx + 1
                break
        for idx in range(token.idx + 1, self.end_idx + 1):
            if self.string[idx] not in string.ascii_letters + "'’-":
                end = idx
                break
        return self.string[start:end]

    def is_contraction(self, token):
        """returns boolean value indicating whether @token has a contraction"""
        for idx in range(token.idx, self.end_idx + 1):
            if self.string[idx] in ["'", "’"]:
                return True
            if re.match("^[\w-]+$", self.string[idx]) is None:
                return False
        return False

    def is_negated(self, token):
        """returns boolean value indicating whether @token is negated"""
        return any([c.dep_ == "neg" for c in token.children])

    def capitalize(self, string):
        """takes a word array @string and capitalizes only the first word in the sentence"""
        if len(string):
            return string[0].capitalize() + string[1:]
        return ""

    def is_capitalized(self, string):
        """returns boolean value indicating whether @string is capitalized"""
        return self.capitalize(string) == string

    def is_modal(self, token):
        """returns boolean value indicating whether @token is a modal verb"""
        if token.tag_ == "MD":
            return True

        start_idx = token.idx
        for m in self.MODALS:
            end_idx = start_idx + len(m)
            if self.string[start_idx:end_idx] == m:
                return True
        return False

    def has_object(self, token):
        """returns boolean value indicating whether @token has an object dependant (child)"""
        return any([c.dep_ in self.OBJECTS for c in token.children])
    
    def has_aux(self, verb):
        """returns boolean value indicating whether @verb has an aux dependant (child)"""
        return  any([c.dep_ == 'aux' for c in verb.children])

    def adjective_synsets(self, synsets):
        """returns a list of all adjectives in the WordNet vocabulary"""
        return [s for s in synsets if s.pos() in {"a"}]

    def is_gradable_adjective(self, word: str):
        """returns boolean value indicating whether @word is a gradable adjective"""
        orig_synsets = set(self.adjective_synsets(wn.synsets(word)))

        superlative = {}
        superlative[0] = word + "est"  # green -> greenest
        superlative[1] = word + word[-1] + "est"  # hot -> hottest
        if len(word) and word[-1] == "e":
            superlative[2] = word[:-1] + "er"  # 'blue' -> 'bluest'
        if len(word) > 1 and word[-2] == "y":
            superlative[3] = word[:-1] + "ier"  # happy -> happiest

        return any(
            [
                len(
                    set(self.adjective_synsets(wn.synsets(sup))).intersection(
                        orig_synsets
                    )
                )
                for sup in superlative.values()
            ]
        )
                
    def surface_contract(self, string):
        """Contract verbs and fix errors in contractions"""
        
        # feature 34 and fixes
        orig = string  # .copy()

        string = re.sub(r"\byou all\b", "y'all", string, flags=re.IGNORECASE)
        string = re.sub(r"\bnn't\b", "n't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bwilln't\b", "won't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bI'm going to\b", "I'ma", string, flags=re.IGNORECASE)
        string = re.sub(r"\bI am going to\b", "I'ma", string, flags=re.IGNORECASE)
        string = re.sub(r"\bI will\b ", "I'ma ", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhave to\b", "gotta", string, flags=re.IGNORECASE)
        string = re.sub(r"\bneed to\b", "gotta", string, flags=re.IGNORECASE)
        string = re.sub(r"\bgot to\b", r"gotta", string, flags=re.IGNORECASE)
        string = re.sub(r"\btrying to\b", "tryna", string, flags=re.IGNORECASE)
        string = re.sub(r"\bbeen been\b", "been", string, flags=re.IGNORECASE)

        if string != orig:
            self.modification_counter["lexical"] += 1
            self.modification_counter["total"] += 1

        return string
                
                
    def remove_determiners(self, token):
        """Convert 'the dog' into just 'dog'"""
        for child in token.children:
            if child.dep_ == "det":
                self.set_rule(child, "")
                
    def null_genitive(self):
        """Removes the possessive s and other possessive morphology"""
        
        # feature 77
        for token in self.tokens:
            if token.tag_ == "POS":
                self.set_rule(token, " ", "null_genetive")  # drop and leave a space
        
    def completive_done(self):
        # feature 104, 105
        self.completive_been_done(p=1.0)

    def completive_been_done(self, p=0.5):
        """Implements completive been/done"""
        for token in self.tokens:
            if token.lower_ in {"done", "been"}:
                # if the verb is done/been, we don't want to duplicate it
                continue
            if token.tag_ == "VBD":
                if any(
                    [
                        child.dep_ == "npadvmod" or child.lemma_ in {"already", "ago"}
                        for child in token.children
                    ]
                ):
                    self.rules[
                        (token.idx - 1, token.idx - 1)  # starting with this token
                    ] = (
                        {"value": " done", "type": "been_done"}
                        if random.random() < p
                        else {"value": " been", "type": "been_done"}
                    )
                    self.modification_counter["been_done"] += 1
                    self.modification_counter["total"] += 1
            elif token.tag_ == "VBN":
                for child in token.children:
                    if (
                        child.dep_ == "aux"
                        and child.lemma_ == "have"
                        and not self.is_contraction(child)
                    ):
                        swap = "done"  # if random.random()<p else 'been' # note: removed 'been' for this case b/c an annotator didn't find it grammatical
                        self.set_rule(child, swap, "been_done")
    
    def surface_future_sub(self, string, replace="finna"):
        """Fix erroneous future tense"""
        
        # feature 126
        orig = string  # .copy()
        string = re.sub("(?:just )?about to", replace, string, flags=re.IGNORECASE)
        string = re.sub(
            "almost (?:going to|gonna|gunna)", replace, string, flags=re.IGNORECASE
        )
        string = re.sub("fixin(?:'|g)? to", replace, string, flags=re.IGNORECASE)

        if string != orig:
            self.modification_counter["lexical"] += 1
            self.modification_counter["total"] += 1
        return string
    
    def negative_concord(self):
        # feature 154
        for token in self.tokens:
            # if the token is a negation of a verbal head then
            if (
                (token.dep_ == "neg") or (token.lower_ in self.NEGATIVES.values())
            ) and (token.head.pos_ in {"VERB", "AUX"}):
                neg = ""
                for val in self.NEGATIVES.values():
                    if token.lower_ == val:
                        neg = val
                        break
                # consider special ain't for non-contractions
                if (token.head.lemma_ == "be") and not len(
                    set(self.get_full_word(token)).intersection({"'", "’"})
                ):
                    self.set_rule(token.head, "ain't", "negative_concord")
                    self.set_rule(token, neg, "negative_concord")

                for child in token.head.children:
                    # Find the object child and check that it is an indefinite noun
                    if (child.dep_ in self.OBJECTS) and self.is_indefinite_noun(
                        child
                    ):  # \
                        # or ((child.dep_ in {'advmod'}) and (child.pos_ in {'ADV', 'ADJ'})):
                        if str(child) in list(self.NEGATIVES.keys()):
                            # there is a special NPI word that is the negation of this one
                            self.set_rule(
                                child, self.NEGATIVES[str(child)], "negative_concord"
                            )
                        else:
                            # otherwise, just append the prefix "no" to negate the word
                            self.set_rule(child, "no " + str(child), "negative_concord")
                        self.remove_determiners(child)
                    # Next, look to the prepositional phrases and do the same thing
                    elif child.dep_ == "prep":
                        for c in child.children:
                            if (c.dep_ == "pobj") and self.is_indefinite_noun(c):
                                self.set_rule(c, "no " + str(c), "negative_concord")
                                self.remove_determiners(c)
                                
    def surface_aint_sub(self, string):
        """Substitute ain't in the place of other negations"""
        
        # feature 155, 156, 157
        orig = string  # .copy()
        string = re.sub(r"\bgon not\b", "ain't gon", string, flags=re.IGNORECASE)
        string = re.sub(r"\bis no\b", "ain't no", string, flags=re.IGNORECASE)
        string = re.sub(r"\bare no\b", "ain't no", string, flags=re.IGNORECASE)
        string = re.sub(r"\bare no\b", "ain't no", string, flags=re.IGNORECASE)
        string = re.sub(r"\bis not\b", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bisn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bare not\b", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\baren['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhaven['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhasn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bhadn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)
        string = re.sub(r"\bdidn['‘’`]?t", "ain't", string, flags=re.IGNORECASE)

        if string != orig:
            self.modification_counter["lexical"] += 1
            self.modification_counter["total"] += 1
        return string

    def uninflect(self):
        """
        change all present tense verbs to 1st person singular form
        change "were" to "was"
        """
        
        # feature 158, 170
        for token in self.tokens:
            # we don't want to change contractions or irregulars
            if (token.lower_ not in {"am", "is", "are"}) and (
                not self.is_contraction(token)
            ):
                if token.tag_ in {"VBZ", "VBP"}:  # simple present tense verb
                    uninflected = token._.inflect("VBP", form_num=0)
                    if not uninflected:
                        continue
                    if str(token) != str(
                        uninflected
                    ):  # we need to remember this change
                        self.set_rule(token, uninflected, "uninflect")
                elif token.tag_ in {
                    "VBD"
                }:  # simple past tense verb; could also consider past participle (VBN)
                    uninflected = token._.inflect("VBD", form_num=0)
                    if not uninflected:
                        continue
                    if str(token) != str(
                        uninflected
                    ):  # we need to remember this change
                        self.set_rule(token, uninflected, "uninflect")
                            
    def existential_dey_it(self, p=0):
        """Adds the existential dey/it construction"""
        
        # feature 172, 173
        replace = "dey" if random.random() < p else "it"
        for token in self.tokens:
            if (token.dep_ == "expl") and (str(token).lower() == "there"):
                self.set_rule(token, replace, "dey_it")
    
    def drop_aux(self, gonna_filter=False, question_filter=False, progressive_filter=False):
        """drop copulas and other helping verbs"""
        
        # feature 174, 175, 176, 178, 179
        for token in self.tokens:
            
            # handle restrictive case options
            gonna = (token.lower_ in {"gonna", "gunna"}) or (token.lemma_ in {"go", "gon"})
            question = ('?' in {c.lemma_ for c in token.head.children}) or (token.tag_ in {'WDT', 'WP', 'WP$', 'WRB'})
            progressive = token.head.tag_ == 'VBG'
            if (gonna or not gonna_filter) and (question or not question_filter) and (progressive or not progressive_filter): 
                
                if not self.is_contraction(token):  # we don't want to change contractions
                    if token.lemma_ == "be":  # copulas are a separate case
                        if token.lower_ in {"is", "are"}:
                            # we don't want to mess with relative clauses
                            # e.g. "are" in "That's just what you are today"
                            if (
                                not self.is_negated(token)
                                and ("comp" not in token.dep_)
                                and not any(
                                    [c.dep_ in {"ccomp", "expl"} for c in token.children]
                                )
                                and any(
                                    [
                                        c.dep_ in self.OBJECTS.union({"advmod", "acomp"})
                                        for c in token.children
                                    ]
                                )
                            ):  # and (token.dep_ != 'ROOT'):
                                self.set_rule(token, "", "drop_aux")  # drop copula
                        else:
                            pass  # don't change past-tense copula
                    elif (token.pos_ == "AUX") and (token.head.dep_ != "xcomp"):
                        # next, look at other auxilliaries that are not complements
                        if token.dep_ == "aux":
                            if str(token) == "will":
                                self.set_rule(token, "gon", "lexical")  # future tense
                            elif (token.head.lemma_ != "be") and not (
                                self.is_negated(token.head) or self.is_modal(token)
                            ):
                                self.set_rule(token, "", "drop_aux")  # drop
                        elif token.dep_ == "ROOT":
                            if any([child.dep_ in ["acomp"] for child in token.children]):
                                self.set_rule(token, "", "drop_aux")  # drop
        
    def null_relcl(self):
        """Removes the complementizer for relative clauses"""
        
        # feature 193
        for token in self.tokens:
            # relative clause Wh-determiner / Wh-pronoun
            if ((token.head.dep_ in {"relcl"}) or (token.dep_ in {"relcl"})) and (
                token.lemma_ in {"that", "who"}
            ):
                self.set_rule(token, "", "null_relcl")  # drop
    
    def negative_inversion(self):
        # feature 226
        for token in self.tokens:
            consider = None
            if token.lower_ == "no":
                consider = token.head
            elif str(token) in self.NEGATIVES.values():
                consider = token
            else:
                continue

            if not self.is_clause_initial(consider):
                continue

            found_aux = False
            for child in consider.head.children:
                if (child.dep_ == "aux") and (child.lower_ != "to"):
                    found_aux = True
                    if not self.is_contraction(child):
                        self.set_rule(child, "")
                        self.set_rule(
                            token,
                            str(child) + "n't " + token.lower_,
                            "negative_inversion",
                        )

            if not found_aux:
                self.set_rule(token, "don't " + token.lower_)
                            
    def got(self):
        """Changes verbal present-tense 'have' to 'got' (AAVE)"""
        for token in self.tokens:
            if (
                token.lower_ in {"have", "has"}
                and token.pos_ == "VERB"
                and self.has_object(token)
                and not any([c.dep_ == "aux" for c in token.children])
            ):
                self.set_rule(token, "got", "got")
                            
    def ass_pronoun(self, p=0.1):
        """Implements ass camouflage constructions, reflexive constructions, and intensifiers"""
        for token in self.tokens:
            if (token.dep_ in self.OBJECTS) and (
                token.tag_ == "PRP"
            ):  # the token is an object pronoun
                if (
                    str(token) in self.POSSESSIVES
                ):  # the token has a possessive counterpart
                    self.set_rule(token, self.POSSESSIVES[str(token)] + " ass", "ass")
            elif (
                (token.dep_ == "amod")
                and (token.tag_ == "JJ")
                and self.is_gradable_adjective(str(token))
            ):  # the token is a gradable adjective modifier
                if random.random() < p:
                    self.set_rule(token, str(token) + "-ass", "ass")
            elif token.pos_ == "VERB":
                # TODO: handle imperative: "Get inside" => "Get your ass inside"
                pass
            
    ## Surface features and fixes and lexical substitutions

    def surface_dey_conj(self, string):
        """Fix errors in conjucation left by it/dey construction, etc. (AAVE)"""
        string = re.sub(r"\bDey are", "Dey is", string)
        string = re.sub(r"\bAre dey\b", "Is dey", string)
        string = re.sub(r"\bIt are", "It is", string)
        string = re.sub(r"\bAre it", "Is it", string)
        string = re.sub(r"\bdey are", "dey is", string)
        string = re.sub(r"\bare dey\b", "is dey", string)
        string = re.sub(r"\bit are", "it is", string)
        string = re.sub(r"\bare it", "is it", string)
        return string

    def surface_lexical_sub(self, string, p=0.4):
        """Make all lexical substitutions indicated in the dictionary @self.lexical_swaps"""
        for sae in self.lexical_swaps.keys():
            if ("." not in sae) and len(self.lexical_swaps[sae]):
                for match in re.finditer(r"\b%s\b" % sae, string, re.IGNORECASE):
                    if random.random() < p:  # swap
                        start, end = match.span()

                        swap = random.sample(list(self.lexical_swaps[sae]), 1)[0]
                        string = "%s%s%s" % (string[:start], swap, string[end:])

                        self.modification_counter["lexical"] += 1
                        self.modification_counter["total"] += 1
        return string
    
    ## Helper functions


    def get_clause_origin(self, token):
        """Returns the root or parent node in the clause to which @token belongs"""
        t = token
        while ("comp" not in t.dep_) and (t.dep_ not in {"ROOT", "conj"}):
            t = t.head
        return t

    def subtree_min_idx(self, token, use_doc_index=False, use_max_idx=False, exclude_tags={}):
        """Returns the minimum character index or the left side of the clause to which @token belongs"""
        indices = []
        for child in token.children:
            if ("comp" not in child.dep_) and child.dep_ not in {"punct", 'advcl'} and child.tag_ not in exclude_tags:
                indices.append(self.subtree_min_idx(child, use_doc_index=use_doc_index, use_max_idx=use_max_idx))
        if use_doc_index:
            indices.append(token.i)
        else:
            indices.append(token.idx)
        if use_max_idx:
            return max(indices)
        return min(indices)

    def is_clause_initial(self, token):
        """Returns boolean value indicating whether @token is the first token in its clause (useful for negative inversion)"""
        root = self.get_clause_origin(token)
        return token.idx == self.subtree_min_idx(root)

    def is_indefinite_noun(self, token):
        """returns boolean value indicating whether @token is an indefinite noun"""
        # A noun may indefinite if either it has no determiner or it has an indefinite article
        # It cannot be a proper noun, pronoun, possessive, or modifier, nor can it be negated
        return (
            (token.pos_ in {"NOUN"})
            and (token.tag_ not in {"PRP", "NNP"})
            and (str(token) not in self.NEGATIVES.values())
            and not any(
                [c.dep_ in {"poss", "amod", "neg", "advmod"} for c in token.children]
            )
            and not any(
                [
                    (c.dep_ == {"det", "nummod"}) and (c.lemma_ not in {"a", "an"})
                    for c in token.children
                ]
            )
        )         

    def remove_recursive(self, token):
        """Remove @token and all children of @token in the dependency tree"""
        if not token:
            return
        self.set_rule(token, "")
        for child in token.children:
            self.remove_recursive(child)
            
    ## Compile

    def compile_from_rules(self):
        """
        compile all accumulated morphosyntactic rules

        rules will be a dictionary of the following key/value form:

        { (start_idx, end_idx): swap_str }

        """
        prev_idx = 0
        compiled = ""
        for indices in sorted(self.rules.keys()):
            start_idx, end_idx = indices
            if start_idx >= 0:
                compiled += self.string[prev_idx:start_idx]
                prev_idx = end_idx + 1

            compiled += self.rules[indices]["value"]

            if (
                len(self.rules[indices]["value"])
                and (self.rules[indices]["value"] != " ")
                and (prev_idx < len(self.string))
            ):
                compiled += self.string[end_idx]

        if prev_idx < len(self.string):
            compiled += self.string[prev_idx:]
        return compiled

    def highlight_modifications_html(self):
        """Return an HTML highlighting and indexing of all modified tokens (used for MTurk validation)"""
        prev_idx = 0
        compiled = ""
        j = 1
        for indices in sorted(self.rules.keys()):
            start_idx, end_idx = indices
            if start_idx >= 0:
                compiled += self.string[prev_idx:start_idx]
                prev_idx = end_idx + 1

            compiled += (
                ("<a href='%s' title='%s'><mark>" % (self.rules[indices]["type"], j))
                + self.string[start_idx:end_idx]
                + "</mark></a>"
            )
            j += 1

            if (
                len(self.rules[indices]["value"])
                and (self.rules[indices]["value"] != " ")
                and (prev_idx < len(self.string))
            ):
                compiled += self.string[end_idx]

        if prev_idx < len(self.string):
            compiled += self.string[prev_idx:]
        return compiled