from sklearn.pipeline import Pipeline
import joblib
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml


def save_model(features, target, *to_save):
    pl = [(chr(97 + i), val) for i, val in enumerate(to_save)]
    pipeline = Pipeline(pl)
    pipeline.fit(features, target)
    joblib.dump(pipeline, 'output/model.pkl')
    # pipeline = PMMLPipeline(pl)
    # pipeline.fit(features, target)
    # sklearn2pmml(pipeline, "output/model.pmml", with_repr = True)

def load_model(location):
    # 加载模型
    loaded_model = joblib.load(location)
    return loaded_model
