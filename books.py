from roboflow import Roboflow
rf = Roboflow(api_key="6usTHvUiI4fePTEGtQuU")
project = rf.workspace().project("book-fkqrm")
model = project.version(2).model

# infer on a local image
print(model.predict("D:\karma\Code\Python\BooksDetection\Kitap_202108210905523625571.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("D:\karma\Code\Python\BooksDetection\Kitap_202108210905523625571.jpg", confidence=40, overlap=30).save("prediction.jpg")

#infer on an image hosted elsewhere
#print(model.predict("0002031940001-1.jpg", hosted=True, confidence=40, overlap=30).json())