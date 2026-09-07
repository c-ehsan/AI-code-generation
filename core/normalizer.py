from hazm import Normalizer

class ApplyNormalizer:
    def __init__(self,text):
        self.text=text

    def apply_normalizer_func(self):
        normalizer=Normalizer()
        
        self.normalized_text=normalizer.normalize(self.text)
        self.normalized_text=self.normalized_text.strip()
        self.normalized_text=self.normalized_text.split()
        return self.normalized_text

initializer=ApplyNormalizer("سلام خوبی چههههه خبر؟؟؟")
normalized_text=initializer.apply_normalizer_func()
print(normalized_text)


        
