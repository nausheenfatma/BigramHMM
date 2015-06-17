# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:22:13 2015

@author: nausheenfatma
"""

#Bigram HMM 

import shlex #to split by quotes 

class HMM :
    #Global variables
    def __init__(self):
        self.tag_unigrams={}            #contains tags,and their counts
        self.dict_word_tags={}          #contains word_tag pair and their counts
        self.tag_bigrams={}             #contains tag bigrams and its count
        self.tag_trigrams={}            #contains tag trigrams and its count                        
        self.train_file_path="data/Hindi_POS.txt"
        self.test_file_path="data/Hindi_Test.txt"
        self.output_file_path="output/output.txt"
        #self.default_tag="NN"
        self.emission_transition_map={}

    
    def find_word_tag_pairs(self,filename):
        f1=open(filename,"r")
        for line in f1:
            sentence=shlex.split(line.rstrip())[1]
            tokens=sentence.split()
            for token in tokens:
                if not token in self.dict_word_tags:
                    self.dict_word_tags[token]=1
                else :
                    self.dict_word_tags[token]=self.dict_word_tags[token]+1
                
    
    def e(self,x,s):                             #Emission function e(word|tag)
        word_tag=x+"_"+s
        e=0
        if s in  self.tag_unigrams.keys() and word_tag in self.dict_word_tags.keys():
            e=(self.dict_word_tags[word_tag]+1)/ float(self.tag_unigrams[s]+1000)
	else :
	    e=1/float(self.tag_unigrams[s]+1000) #smoothing wowwwwww :)
        return e
     
        
    def q(self,current_tag,previous_tag):         #Transition function q(current_tag|previous_tag)=count(previous tag,current tag)/count(previous tag)
        bigram=previous_tag+" "+current_tag
        unigram=previous_tag
        if bigram in self.tag_bigrams.keys() and unigram in self.tag_unigrams.keys() :
            q_value= self.tag_bigrams[bigram]/float(self.tag_unigrams[unigram])
        else :
            q_value=0
        return q_value 

        
    def find_ngrams(self,filename,n):
        f1=open(filename,"r")
        ngrams={}
        for line in f1:
            sentence=shlex.split(line.rstrip())[1]
            tokens=sentence.split()
            tags=[]
            for i in range(n) :
                tags.append('*')
            for token in tokens:
                wordtag=token.split("_")
                tags.append(wordtag[1])
            
            for i in range(len(tags)+1-n):
                gram=""
                for j in range(n):
                    gram=gram+tags[i+j]
                    if j != n-1:
                        gram=gram+" "
                        
                if gram not in ngrams:
                    ngrams[gram]=1
                else :
                    ngrams[gram]=ngrams[gram]+1
        f1.close()
        return ngrams    
        

            
    def add_entry_to_map(self,word,word_index,max_prod,candidate_previous_tag,current_tag):

        if word_index not in self.emission_transition_map.keys() :
            self.emission_transition_map[word_index]={} 
        self.emission_transition_map[word_index][current_tag]=[max_prod,candidate_previous_tag]
     
        
    def previous_tag_product(self,word_index,tag) :
        product_value=0
        index_value=word_index-1
        if index_value ==-1 :
            product_value=1
        elif index_value in self.emission_transition_map.keys() and tag in self.emission_transition_map[index_value].keys():
           product_value=self.emission_transition_map[index_value][tag][0]
        return product_value        
        
    def predict_test_tags(self,filename):
        testfile=open(filename,"r")
        outputfile=open(self.output_file_path,"a")
        
        for line in testfile:
            self.emission_transition_map={}
            words=line.rstrip().split()
            i=0
            for i in range(len(words)):
                if i==0:
                    self.set_first_word_of_sentence(words[i]) #previous tag for first word is '*'
                else :
                   self.find_tag_by_max_product(words[i],i)   #check for all tags
                i=i+1 
            self.find_tag_for_sentence(words,outputfile)   

    def set_first_word_of_sentence(self,word):
        candidate_previous_tag="*"
        max_prod=0
        for tag in self.tag_unigrams.keys() :		      # for first word for all tags
            if tag != '*' :
			    #print "emission first word",self.e(word,tag)
                            product=self.e(word,tag)*self.q(tag,candidate_previous_tag)  
			    #print "product first word",product                  
                            if product>=max_prod:
                                max_prod=product
                            if product > 0:                  # save only those tags whose product >0
                                self.add_entry_to_map(word,0,product,"*",tag)
    
    
    def find_tag_by_max_product(self,word,word_index):
        #print "word",word
       
        for i in range(len(self.tag_unigrams)) :            #for all current tags find the product value
                current_tag=(self.tag_unigrams).keys()[i]
		#print "current_tag",current_tag
                max_prod=0
                previous_tag=""
                emission_value=self.e(word,current_tag)
		#print "emission_value",emission_value
                if emission_value!=0 :
                    for j in range(len(self.tag_unigrams)) : # for all candidate previous tags,check for maximum product 
                            candidate_previous_tag=(self.tag_unigrams).keys()[j]
			   # print self.emission_transition_map
                            if candidate_previous_tag in self.emission_transition_map[word_index-1].keys():
                                product=self.q(current_tag,candidate_previous_tag) * self.previous_tag_product(word_index,candidate_previous_tag)                    
                                if product>=max_prod :      # finding the maximum product of emission-transmission-previous_candidate_tag product
                                    max_prod=product
                                    previous_tag=candidate_previous_tag
                    total_product=max_prod*emission_value
                    if total_product >0 :                   #save into map only if the product in non-zero,otherwise successive products would also become zero,no point saving them
                        self.add_entry_to_map(word,word_index,total_product,previous_tag,current_tag) 
                
            
                    
        
        
    def find_tag_for_sentence(self,words,outputfile) :
        no_of_words=len(self.emission_transition_map)
        j=no_of_words-1
        max_prod=0
        word_tags={}
        tag_predicted="&"

        for tag in self.emission_transition_map[j].keys():  # finding the max product in for the last word
                prod=self.emission_transition_map[j][tag][0]
                if prod>max_prod :
                    max_prod=prod
                    tag_predicted=tag
                    previous_tag=self.emission_transition_map[j][tag][1]
        word_tags[j]=tag_predicted
        previous_tag=tag_predicted
                
        
        while j>0 :                                         #retracing from the last word to first word
            tag_predicted=self.emission_transition_map[j][previous_tag][1]
            previous_tag=tag_predicted
            word_tags[j-1]=tag_predicted   
            j=j-1       
        for j in range(no_of_words) :                       #writing the words and tags in file from 1st to last word
           # print "word:",words[j]
           # print "tag",word_tags[j]
            word_tag=words[j]+"_"+word_tags[j]   
            outputfile.write(word_tag+" ")
        outputfile.write("\n")   
        
                           
def main():
    hmm_object=HMM()
    hmm_object.find_word_tag_pairs(hmm_object.train_file_path)
    hmm_object.tag_unigrams=hmm_object.find_ngrams(hmm_object.train_file_path,1)
    hmm_object.tag_bigrams=hmm_object.find_ngrams(hmm_object.train_file_path,2)
    hmm_object.predict_test_tags(hmm_object.test_file_path)   

if __name__ == "__main__": main()
            
        

    
