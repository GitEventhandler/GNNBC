hp_cora = {
    "wd1": 4e-2,
    "wd2": 2e-5,
    "lambda_1": 1.5,
    "lambda_2": 0.5,
    "layer": 2,
    "dropout": 0.35,
    "lr": 0.03,
}

hp_pubmed = {
    "wd1": 4e-2,
    "wd2": 2e-5,
    "lambda_1": 1.5,
    "lambda_2": 0.5,
    "layer": 2,
    "dropout": 0.35,
    "lr": 0.03,
}

hp_citeseer = {
    "wd1": 5e-2,
    "wd2": 2e-4,
    "lambda_1": 4.0,
    "lambda_2": 1.5,
    "layer": 4,
    "dropout": 0.15,
    "lr": 0.015,
}

hp_chameleon = {
    "wd1": 8e-2,
    "wd2": 0.0,
    "lambda_1": 3.0,
    "lambda_2": 1.0,
    "layer": 6,
    "dropout": 2e-2,
    "lr": 0.015,
}

hp_squirrel = {
    "wd1": 0.075,
    "wd2": 0.0,
    "lambda_1": 3.5,
    "lambda_2": 3.0,
    "layer": 2,
    "dropout": 0.0,
    "lr": 0.005,
}

hp_amazon_computers = {
    "wd1": 5e-4,
    "wd2": 0.0,
    "lambda_1": 1.0,
    "lambda_2": 1.0,
    "layer": 2,
    "dropout": 0.05,
    "lr": 0.03,
}

hp_amazon_photo = {
    "wd1": 5e-4,
    "wd2": 0.0,
    "lambda_1": 1.0,
    "lambda_2": 1.0,
    "layer": 2,
    "dropout": 0.05,
    "lr": 0.03,
}


def get_hyper_param(name: str):
    name = name.lower()
    if name == "cora":
        return hp_cora
    elif name == "pubmed":
        return hp_pubmed
    elif name == "citeseer":
        return hp_citeseer
    elif name == "chameleon":
        return hp_chameleon
    elif name == "squirrel":
        return hp_squirrel
    elif name == "computers":
        return hp_amazon_computers
    elif name == "photo":
        return hp_amazon_photo
    else:
        raise Exception("Not available")
