class dataManager():

    def __init__(self):

        self.encoderMap={}
        self.decoderMap={}
        self.text=None

    def dataIngestion(self):

        with  open("D:/INeuron/miniGPT/chef.txt", "r") as file:

            lines = file.readlines()
            text = " ".join(lines)
            text = text.split("\n")
            self.text = " ".join(text)

    def createMap(self):

        uniqueElements = list(set(self.text))
        uniqueElements.sort()
        self.encoderMap = {i: uniqueElements.index(i) for i in uniqueElements}
        self.decoderMap = {uniqueElements.index(i): i for i in uniqueElements}

    def tokenizer(self,string):

        array = []
        for i in string:
            array.append(self.encoderMap[i])

        return array


    def detokenizer(self,array):
        string = ""
        for i in array:
            string += self.decoderMap[i]
        return string