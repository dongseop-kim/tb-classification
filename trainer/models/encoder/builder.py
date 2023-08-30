import timm


def build_encoder(name: str, **kwargs):
    timm_list = timm.list_models(pretrained=True)
    if name in timm_list:
        return timm.create_model(model_name=name, pretrained=True, **kwargs)
    else:
        raise KeyError(f"'{name}' is not in backbone list")
