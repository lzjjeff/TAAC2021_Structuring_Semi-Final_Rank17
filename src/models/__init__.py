from src.models.sence_seg import SceneSegCNN, SceneSegMultiCNN, SceneSegCNNTRM, SceneSegTransformer
from src.models.tagging import MultiTagging, MultiTaggingWin


SceneSegMODEL = {"cnn": SceneSegCNN,
                 "multicnn": SceneSegMultiCNN,
                 "cnntrm": SceneSegCNNTRM,
                 "transformer": SceneSegTransformer,
                 }
TaggingMODEL = {"nextvlad": MultiTagging,
                "win": MultiTaggingWin,}