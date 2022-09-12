from .BaseDialect import BaseDialect

class AfricanAmericanVernacular(BaseDialect):
    def __init__(self, lexical_swaps={}, morphosyntax=True):
        super().__init__(lexical_swaps, morphosyntax)
        self.dialect_name = "AAVE Associated Transformations"
        self.surface_subs = [
            self.surface_contract,
            self.surface_dey_conj,
            self.surface_aint_sub,
            self.surface_future_sub,
            self.surface_lexical_sub
        ]
        self.morphosyntax_transforms = [
            self.uninflect,
            self.drop_aux,
            self.negative_concord,
            self.existential_dey_it,
            # self.ass_pronoun,
            self.null_genitive,
            self.null_relcl,
            self.negative_inversion,
            self.completive_been_done,
            self.got
        ]
        
    def clear(self): 
        """clear all memory of previous string"""
        self.string = ""
        self.rules = {}
        self.doc = None
        self.tokens = []
        self.end_idx = 0
        self.modification_counter = {
            label: 0 for label in ['uninflect', 'lexical', 'drop_aux', 'dey_it', 'negative_concord', 
            'ass', 'null_genetive', 'null_relcl', 'negative_inversion', 'been_done', 'got', 'total']
        }