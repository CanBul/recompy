import numpy

class NearestNeighbour():
    
    def __init__(self,data):
        
        self.data = data
        self.pmin = np.min(data[:,2])
        self.pmax = np.max(data[:,2])
        
    @staticmethod
    def __calculate_similarity(data, new_user):
        
        unique_user_ids = np.unique(data[:,0])
        similarities = []
        threshold = np.ceil(np.sqrt(len(new_user.keys()))) # min item required to calculate similarity
        new_user_items = list(new_user.keys())
        
        for uid in unique_user_ids:
            user_item_ratings = data[data[:,0]==uid][:,[1,2]]
            intersected_items_with_new_users = list(set(new_user_items) & set(user_item_ratings[:,0]))
            
            intersected_uid_ratings = user_item_ratings[np.isin(user_item_ratings[:,0], intersected_items_with_new_users)]
            intersected_new_uid = np.array(list(new_user.items()))
            intersected_new_uid = intersected_new_uid[np.isin(intersected_new_uid[:,0],intersected_items_with_new_users)]

            intersected_uid_ratings = intersected_uid_ratings[intersected_uid_ratings[:,0].argsort()]
            intersected_new_uid = intersected_new_uid[intersected_new_uid[:,0].argsort()]

            if intersected_new_uid.shape[0] >= threshold:
                #mse
                similarity_with_new_user = (np.square(intersected_uid_ratings[:,1] - intersected_new_uid[:,1])).mean(axis=None)
                similarities.append([uid,similarity_with_new_user])
            
        similarities = np.array(similarities)

        return similarities[similarities[:,1].argsort()]
    
                
    def __random_uniform(self):
        return np.random.uniform(self.pmin,self.pmax, self.n).reshape(-1,1)
    
    def __global_mean(self):
        return (np.array([np.mean(self.data[:,2])]*self.n)).reshape(-1,1)
    
    # user rating mean ile item rating meani mean yapip basiyor
    def __mean_of_means(self): 
        
        mean_of_means = np.concatenate((self.__item_mean(),self.__user_mean()),axis=1)       
        return np.mean(mean_of_means,axis=1).reshape(-1,1)
    
    def __item_mean(self):
        
        return self.item_means_matrix[np.isin(self.item_means_matrix[:,0],self.to_be_filled_items)][:,1].reshape(-1,1)         
            
    def __user_mean(self):
        user_mean = np.mean(self.user_items_rating[:,1])
        return np.array([user_mean]*self.n).reshape(-1,1)
                            

    def fit(self,method = None):

        if method == 'random_uniform':
            fill = self.__random_uniform
            
        elif method == 'global_mean':
            fill = self.__global_mean
            
        elif method == 'mean_of_means':            
            fill = self.__mean_of_means
            n = np.unique(self.data[:,1])
            self.item_means_matrix = np.array( [[i,np.mean(self.data[self.data[:,1]==i,2])] for i in n] ) # groupby item rating mean
        
        elif method == 'item_mean':
            fill = self.__item_mean
            n = np.unique(self.data[:,1])
            self.item_means_matrix = np.array( [[i,np.mean(self.data[self.data[:,1]==i,2])] for i in n] ) # groupby item rating mean
        
        elif method == 'user_mean':
            fill = self.__user_mean
        
        elif method == None:
            pass
       
        else:
            raise ValueError('Invalid method. Available techniques are ["item_mean","user_mean","random_uniform","global_mean","mean_of_means"].')

        # fillna loop
        if method != None:

            items = np.unique(self.data[:,1])
            users = np.unique(self.data[:,0])

            for i in users:
                self.user_items_rating = self.data[self.data[:,0]==i][:,[1,2]] # item rating matrix for specified user
                self.to_be_filled_items = items[~np.isin(items,self.user_items_rating[:,0])].reshape(-1,1) 
                self.n = len(self.to_be_filled_items)
                ratings_filled = fill()
                user_filled = np.array([i]*self.n).reshape(-1,1)
                filled_matrix = np.concatenate((user_filled,self.to_be_filled_items,ratings_filled),axis=1)
                self.data = np.append(self.data, filled_matrix,axis=0)
            
    
    def get_recommendations_for_new_user(self,new_user,item_count=5):
        self.new_user = new_user
        self.to_be_excluded_items = list(self.new_user.keys())
        
        
        
        # get similar users
        self.sim = NearestNeighbour.__calculate_similarity(self.data, self.new_user)
        
        
        max_rating = self.data[:,2].max()
        rec_items = []
        
        # select from most similar users till it reaches 5
        for i in self.sim[:,0]:
            
            rec_items.extend(self.data[(self.data[:,0]==i) & ((self.data[:,2] == max_rating))][:,1])
            rec_items = list(set(rec_items) - set(self.to_be_excluded_items)) # exclude all items rated by new user
            
            if len(rec_items) >= item_count:
                rec_items = rec_items[:item_count]
                break
        
        return rec_items
        