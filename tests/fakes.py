from reflexlayer.types import LLMResult, S1Prediction


class FakeS1:
    def __init__(self, choice="a", p=0.95, fail=None):
        self.choice, self.p, self.fail, self.calls = choice, p, fail, 0

    def predict(self, question, options, context):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return S1Prediction(self.choice, {self.choice: self.p}, self.p, 0.0001, 3.0)


class FakeLLM:
    def __init__(self, choice="a", fail=None):
        self.choice, self.fail, self.calls = choice, fail, 0

    def decide(self, question, options, context):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return LLMResult(self.choice, 0.01, 400.0)


class FixedRng:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value
