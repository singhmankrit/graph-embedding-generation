'''Word2Vec : Skip-Gram Implementation (Skip-Gram ie. Guess context words from target word)'''
import argparse
import numpy as np

def parser():
    parser = argparse.ArgumentParser(description="Arguments for Word2Vec (Skip-gram)")
    parser.add_argument('--input_corpus', nargs="?", default="text/test_corpus.txt", 
                        help="Specifying corpus file for input")
    parser.add_argument('--window_size', type=int, default=2, 
                        help="Window Size for context. Default is 2")
    parser.add_argument('--dim', type=int, default=10,
                        help="Dimensions for word embedding/size of hidden layer. Default is 10")
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help="Learning Rate for word2vec. Default is 0.1")
    parser.add_argument('--epochs', type=int, default=100, 
                        help = "Number of epochs for training. Default is 100")
    return parser.parse_args()

#########################################################################################################
class word2vec():
    def __init__(self, args):
        self.size = args.window_size
        self.dim = args.dim
        self.rate = args.learning_rate
        self.epochs = args.epochs

    def generate_train_data(self, corpus):
        vocab = [] # list of unique words in corpus
        for word in corpus:
            word = word.lower()
            if word not in vocab:
                vocab.append(word)
        self.V = len(vocab) # stores number of unique words
        self.w_i = dict((word, i) for i, word in enumerate(vocab))
        
        # creating training data format using skip-gram
        train_data = []
        for i, word in enumerate (corpus):
            instance = self.one_hot(word.lower())
            context = []
            for j in range (i-self.size, i+self.size+1):
                if i!=j and j>=0 and j<len(corpus):
                    rep = self.one_hot(corpus[j].lower())
                    context.append(rep)
            train_data.append([instance, context])
        return np.array(train_data, dtype=object)
        
    def one_hot (self, word):
        vector = np.zeros(self.V)
        vector[self.w_i[word]] = 1
        return vector
    
    def train(self, train_data):
        self.w1 = np.random.uniform(-1, 1, (self.V, self.dim))
        self.w2 = np.random.uniform(-1, 1, (self.dim, self.V))
        for i in range (self.epochs):
            self.loss = 0
            for instance, context in train_data:
                predict, hidden, output = self.forward_pass(instance)
                EI = np.sum([np.subtract(predict, word) for word in context], axis=0) # Error calculation
                self.backprop(EI, hidden, instance)
                
                self.loss -= sum(output[np.where(word==1)] for word in context)
                self.loss += len(context) * np.log(np.sum(np.exp(output)))
            print("Epoch: ", i, " Loss: ", self.loss)
                
    def forward_pass(self, instance):
        hidden = np.dot(self.w1.T, instance)
        output = np.dot(self.w2.T, hidden)
        predict = self.softmax(output)
        return predict, hidden, output
    
    def softmax(self, output):
        # softmax
        vec = np.exp(output - np.max(output))
        return vec/vec.sum(axis=0)
    
    def backprop(self, EI, hidden, instance):
        # computing weights to be adjusted
        dl_dw2 = np.outer(hidden, EI)
        dl_dw1 = np.outer(instance, np.dot(self.w2, EI.T))
        """print('Instance Dimension: ', instance.shape)
        print('Self W2 Dimension: ', self.w2.shape)
        print('Error Dimension: ', EI.T.shape)
        print('D1 Dimension: ', dl_dw1.shape)"""
        # Update weights
        self.w1 = self.w1 - (self.rate * dl_dw1)
        self.w2 = self.w2 - (self.rate * dl_dw2)

    def word_vec(self, word):
        word = word.lower()
        return self.w1[self.w_i[word]]


#########################################################################################################
def preprocess(corpus):
    # opening file
    f = open(corpus, 'r')
    contents = f.read()
    f.close()
    # removing punctions
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in contents:
            if char in punc:
                contents = contents.replace(char, " ")
    # creating list of words from the corpus
    words = contents.split()
    return words, contents

def main(args):
    words, sentences = preprocess(args.input_corpus)
    model = word2vec(args)
    training_data = model.generate_train_data(words)
    model.train(training_data)
    print(model.word_vec("hello"))
    

if __name__ == "__main__":
    args = parser()
    main(args)
