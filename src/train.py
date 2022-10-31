from src import model


def train(xTrain, yTrain,xTest,yTest,vocabCount,dim):
    mod = model.network_model(vocabCount,dim)
    history = mod.fit(xTrain, yTrain, epochs=10,validation_data=(xTest, yTest), shuffle=True)
    mod.save('./model/text/')
    return history
