import numpy as np
import sklearn.svm as svm
import time
from sklearn.metrics import accuracy_score,f1_score, precision_score,recall_score,roc_auc_score
import math
KNN_prediction_scores={}
KNN_f1_iter_scores={}
pred_iter=[]
f1_iter=[]
validaation_f1_10=[]
validaation_f1_20=[]
validaation_f1_30=[]
validaation_f1_40=[]
validaation_f1_50=[]
validaation_f1_75=[]

zeta_total=[]
all_margin=[]
obj_fun=[]
reg=[]
prev=[]
weights=[]
intercept=[]
best_iter_index= 100
# query_per_iter=2
class TransductiveSVM(svm.SVC): 
    def __init__(self,kernel="linear",Cl=1,Cu=0.01,gamma=0.1,X2=None,maxiter=-1):
        '''
        Initial TSVM
        Parameters
        ----------
        kernel: kernel of svm
        Cl: Penalty Inductive SVM
        Cu: Penalty Unlabeled set
        gamma: gamma for rbf kernel
        X2: Unlabeled set(only features)
                np.array, shape:[n2, m], n2: numbers of samples without labels, m: numbers of features


        '''

        self.maxiter=maxiter

        self.Cl=Cl
        self.Cu=Cu
        self.kernel = kernel
        self.gamma=gamma
        self.clf=svm.SVC(C=self.Cl,kernel=kernel,gamma=self.gamma,probability=False,max_iter=self.maxiter)
        self.Yresult=None
        self.X2=X2



    def fit(self, X1, Y1,Y_True,max_queries,query_per_iter,positive_mean_at0,neg_mean_at0,positive_mean_at1,neg_mean_at1):
        '''
        Train TSVM by X1, Y1, X2(X2 is passed on init)
        Parameters
        ----------
        X1: Input data with labels
                np.array, shape:[n1, m], n1: numbers of samples with labels, m: numbers of features
        Y1: labels of X1
                np.array, shape:[n1, ], n1: numbers of samples with labels

        '''


        t=time.time()
        X2=self.X2 # ==> Unlabeled set(only features)
        Y1[Y1!=+1]=-1  #==> changing Y1 labels of X1 to -1 when it is = 0
        Y1 = np.expand_dims(Y1, 1) # ==> changing Y1 to a single column vector
        ratio=sum(1 for i in Y1 if i==+1)/len(X1) #==> the ratio of the positive examples
        print("ratio=",ratio)
        num_plus=int(len(X2)* ratio) # ==>number of expected positive examples in X2 acording to the ratio

        N = len(X1) + len(X2)  # ==> the total number of examples in train and test

        sample_weight = np.zeros(N) # ==> initializing weight to be the same # of examples and set them = zeros
        sample_weight[:len(X1)] = self.Cl # ==>set the first values of sample weight (belongs to labeld examples)= the labeled examples penelty, Each C[i] is C[i] = C * sample_weight[i]
        #print(sample_weight)



        # ==> Classify the test examples using < ~w; b >.
        # ==> The num+ test examples with the highest value of (~w *~x_j+ b) are assigned to the class + (y*_j := 1
        # ==> the remaining test examples are assigned to class - (y*_j := -1).


        self.clf.fit(X1,Y1,sample_weight=sample_weight[:len(X1)]) # ==> classifing the labeled examples with normal SVC,  #classify the num_plus examples with the highest value with +1, other -1
        Y2=np.full(shape=self.clf.predict(X2).shape,fill_value=-1) # ==>  y2 =contains the valaue of -1 for all unlabeled examples (X2)
        Y2_d = self.clf.decision_function(X2) # ==>the same shape of (X2) decision function that tells us how close to the line we are (close to the boundary means a low-confidence decision).


        index=Y2_d.argsort()[-num_plus:][::-1] # ==> returns the indexes of the input in the order that would sort the array. takes only the index of +ve examples (ordered from big to small)

        for item in index: # ==> indecies of 140,the # of +ve examples that should be in X2 according to ratio of + examples in X1
            Y2[item]=+1    # ==> set 140 labels (# of expected positve according to the ratio of + in X1) to be equal to 1



        #INIT CMINUS E CLUS
        #C_minus=.00001
        C_minus=.00001
        C_plus=.00001*(num_plus/(len(X2)-num_plus))
        for i in range(len(Y2)):
            if(Y2[i]==+1):
                sample_weight[len(X1)+i]=C_plus
            else:
                sample_weight[len(X1)+i]=C_minus


        global X3
        global Y3
        global y1_len
        global atomic_f1
        global seq_feat
        # global pos_m_1
        # global neg_m_1



        Y2 = np.expand_dims(Y2, 1)
        X3 = np.vstack((X1, X2)) # ==> the vstack() function is used to stack arrays in rows (row wise). stack x1 then x2 in rows
        Y3 = np.vstack((Y1, Y2)) #==> now y3 is a single column array contains y1 rows then y3 rows
        k=0
        atomic_f1=X1[:,11:]
        seq_feat=X1[:,:10]

        y1_len=len(Y1)
        counter=0



        while (C_minus<self.Cu or C_plus<self.Cu): #LOOP 1



            self.clf.fit(X3, Y3, sample_weight=sample_weight) # ==> fitting the whole data  TSVM
            Y3 = Y3.reshape(-1) #==> now y3 is a single row
            #slack=Y3*(self.clf.decision_function(X3))
            slack = Y3*self.clf.decision_function(X3) #==> how far each point in X3 from the margin multiplied by the sign


            idx=np.argwhere(slack<1) # ==> contains the indices of negative slack. argwhere()used to find the indices of array elements that are non-zero, grouped by element.
            eslackD=np.zeros(shape=slack.shape)
            for index in idx:
                eslackD[index]=1-slack[index]
            eslack2=np.zeros(shape=Y2.shape)
            eslack=eslackD[:len(X1)] #EPSILON OF LABELLED DATA
            eslack2=eslackD[len(X1):] # EPSILON FOR UNLABELED DATA

            # Merging eslack2 and Y2 to get the positve and negative eslack2 (required later to calculate the objective function)
            eslack2_with_label=np.concatenate([eslack2.reshape(-1,1),Y2.reshape(-1,1)],axis=1)
            eslack2_positve=np.where(eslack2_with_label[:,1]==+1)
            eslack2_negative=np.where(eslack2_with_label[:,1]==-1)


            condition=self.checkCondition(Y2,eslack2) #CONDITION OF LOOP
            l=0
            # print("condition =================")

            #calculating sum of slack i + slack j

            #=======> predicting outsid loop <===========#
            weights.append(self.clf.coef_)
            intercept.append(self.clf.intercept_)

            #Y_True[Y_True==0]=-1
            pred_outer=self.clf.predict(X2)

            # calculating the margine


            margin_norm=(np.linalg.norm(self.clf.coef_))
            margin_out=2/margin_norm
            obj_out=(margin_norm/2)+(self.Cl*np.sum(eslack))+C_plus*np.sum(eslack2[eslack2_positve])+C_minus*np.sum(eslack2[eslack2_negative])

            obj_fun.append(obj_out)
            all_margin.append(margin_out)

            accuracy_outer=accuracy_score(Y_True,pred_outer)
            f1_outer=f1_score(Y_True,pred_outer)
            print("outer iter(",k,'):',accuracy_outer)
            print("outer iter(",k,') F1:',f1_outer)
            pred_iter.append(accuracy_outer)
            f1_iter.append(f1_outer)
            #=======> ------------------  <===========#
            X1L=np.concatenate([X1,Y1],axis=1)
            #positive and negative training examples indices
            pos_X1_index=np.where(X1L[:,12]==+1)
            neg_X1_index=np.where(X1L[:,12]!=+1)
            counter+=1


            while(condition): #LOOP 2


                # Merging eslack2 and Y2 to get the positve and negative eslack2 (required later to calculate the objective function)
                eslack2_with_label=np.concatenate([eslack2.reshape(-1,1),Y2.reshape(-1,1)],axis=1)
                eslack2_positve=np.where(eslack2_with_label[:,1]==+1)
                eslack2_negative=np.where(eslack2_with_label[:,1]==-1)

                # print("inside @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                l+=1

                i,j=self.getIndexCondition(Y2,eslack2)  #TAKE A POSITIVE AND NEGATIVE SET
                #print("Switching at loop "+str(k)+"."+str(l)+"     index: "+str(i)+" "+str(j))
                #print("Switching values: "+str(eslack2[i])+" "+str(eslack2[j]))
                print('label switching phase:\n value of the mean_at_0 feature befor switching=',X2[i,10])
                print('label Y2[',i,'] befoe switching=',Y2[i])
                print('label Y2[',j,'] befoe switching=',Y2[j])
                Y2[i]=Y2[i]*-1 #SWITCHING EXAMPLE
                Y2[j]= Y2[j]*-1


                # KNN
                #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Choose a specific point
                specific_point1 = X2[i]
                specific_point2 = X2[j]
                knn=20


                #calculating the distance from a data point to the +ve nearest neighbors training examples
                distances_to_pos_point_1 = np.linalg.norm(X1[pos_X1_index] - specific_point1, axis=1)
                distances_to_pos_point_2 = np.linalg.norm(X1[pos_X1_index] - specific_point2, axis=1)


                #calculating the distance from a data point to the -ve nearest neighbors training examples
                distances_to_neg_point_1 = np.linalg.norm(X1[neg_X1_index] - specific_point1, axis=1)
                distances_to_neg_point_2 = np.linalg.norm(X1[neg_X1_index] - specific_point2, axis=1)


                if Y2[i]==+1:


                  # Find the indices of the 10 nearest +ve neighbors
                  indices_to_nearestN_pos_point1 = np.argsort(distances_to_pos_point_1)[:knn]

                  # Get the mean of the nearest neighbors
                  nearest_neighbors_mean_point1_at0 = X1[indices_to_nearestN_pos_point1][:,10].mean()
                  nearest_neighbors_mean_point1_at1=X1[indices_to_nearestN_pos_point1][:,11].mean()

                else:

                  # Find the indices of the 10 nearest -ve neighbors
                  indices_to_nearestN_neg_point1 = np.argsort(distances_to_neg_point_1)[:knn]


                  # Get the mean of the nearest neighbors
                  nearest_neighbors_mean_point1_at0 = X1[indices_to_nearestN_neg_point1][:,10].mean()
                  nearest_neighbors_mean_point1_at1=X1[indices_to_nearestN_neg_point1][:,11].mean()


                if Y2[j]==+1:


                  # Find the indices of the 10 nearest +ve neighbors
                  indices_to_nearestN_pos_point2 = np.argsort(distances_to_pos_point_2)[:knn]

                  # Get the mean of the nearest neighbors
                  nearest_neighbors_mean_point2_at0 = X1[indices_to_nearestN_pos_point2][:,10].mean()
                  nearest_neighbors_mean_point2_at1 = X1[indices_to_nearestN_pos_point2][:,11].mean()

                else:

                  # Find the indices of the 10 nearest -ve neighbors
                  indices_to_nearestN_neg_point2 = np.argsort(distances_to_neg_point_2)[:knn]

                  # Get the mean of the nearest neighbors
                  nearest_neighbors_mean_point2_at0 = X1[indices_to_nearestN_neg_point2][:,10].mean()
                  nearest_neighbors_mean_point2_at1 = X1[indices_to_nearestN_neg_point2][:,11].mean()

                  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^





                # Updating the mean atomic features by the mean of the neerest neighbores
                X2[i,10]=nearest_neighbors_mean_point1_at0
                X2[i,11]=nearest_neighbors_mean_point1_at1

                X2[j,10]=nearest_neighbors_mean_point2_at0
                X2[j,11]=nearest_neighbors_mean_point2_at1



                print('value of the mean_at_0 feature after switching=',X2[i,10])
                print('label Y2[',i,'] after switching=',Y2[i])
                print('label Y2[',j,'] after switching=',Y2[j])

                print('y1_shape=',Y1.shape)
                print('y2_shape=',Y2.shape)

                #update X3 with the new X2 values
                X4 = np.vstack((X1, X2))
                X3=np.copy(X4)
                print ("X3 shape",X3.shape)
                # start_ind=X3.shape[0]-1+i
                # print("x3 value, mean_at_feature",X3[start_ind,10])


                sample_weight[len(X1)+i],sample_weight[len(X1)+j]=sample_weight[len(X1)+j],sample_weight[len(X1)+i] #UPDATE THE WEIGHT
                print('y1_shape=',Y1.shape)
                print('y2_shape=',Y2.shape)

                Y3=np.concatenate((Y1,Y2),axis=0)
                # print('.....')
                self.clf.fit(X3,Y3, sample_weight=sample_weight) #TRAINING WITH NEW LABELLING

                if( counter>best_iter_index):


                  #Active Learning
                  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                  #caclulating the decison function to pick the nearest points to the magrin

                  decision_values = self.clf.decision_function(X2)
                  uncertainty_scores = np.abs(decision_values)
                  top_indices = np.argsort(uncertainty_scores)[:query_per_iter]

                  # Query the true labels for the selected instances
                  new_X = X2[top_indices]
                  new_y = Y_True[top_indices]
                  new_y=new_y.reshape(-1,1)

                  # Add the selected instances to the labeled data
                  X1 = np.concatenate([X1, new_X])
                  Y1 = np.concatenate([Y1, new_y])

                  # Remove the selected instances from the unlabeled data
                  X2 = np.delete(X2, top_indices, axis=0)
                  Y2 = np.delete(Y2, top_indices, axis=0)
                  Y_True = np.delete(Y_True, top_indices, axis=0)

                  print('>>>>>>>>>>>>')

                  #updating X3,Y3
                  X3=np.vstack((X1,X2))
                  Y3=np.vstack((Y1,Y2))
                #   print('>>>>>>>>>>>>')


                  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





                Y3 = Y3.reshape(-1)

                # print('.....')

                #slack =Y3*(self.clf.decision_function(X3))
                slack = Y3*self.clf.decision_function(X3)
                idx = np.argwhere(slack < 1)
                eslackD = np.zeros(shape=slack.shape)

                for index in idx:
                    eslackD[index] = 1 - slack[index]

                eslack = eslackD[:len(X1)]
                eslack2 = np.zeros(shape=Y2.shape)
                eslack2 = eslackD[len(X1):]
                condition = self.checkCondition(Y2, eslack2)
                #=======> predicting inside loop <===========#

                weights.append(self.clf.coef_)
                intercept.append(self.clf.intercept_)

                #Y_True[Y_True==0]=-1
                pred_inner_active=self.clf.predict(X2)
                accuracy_inner=accuracy_score(Y_True,pred_inner_active)
                f1_inner=f1_score(Y_True,pred_inner_active)
                KNN_prediction_scores[l]=pred_inner_active

                print("inner iter(",l,') accuracy:',accuracy_inner)
                print("inner iter(",l,') F1:',f1_inner)
                print("precision:",precision_score(Y_True,pred_inner_active))
                print("recall:",recall_score(Y_True,pred_inner_active))
                print("Roc:",roc_auc_score(Y_True,pred_inner_active))






                pred_iter.append(accuracy_inner)
                f1_iter.append(f1_inner)

                # calculating the margine


                margin_norm=(np.linalg.norm(self.clf.coef_))
                margin_in=2/margin_norm
                #obj_in=(margin_norm/2)+(self.Cl*np.sum(eslack))+C_plus*np.sum(eslack2[eslack2_positve])+C_minus*np.sum(eslack2[eslack2_negative])
                prev_max=(2/margin_norm)+(self.Cl*np.sum(eslack))+C_plus*np.sum(eslack2)
                prev.append(prev_max)

                reg_in= (self.Cl*np.sum(eslack))+C_plus*np.sum(eslack2)
                reg.append(reg_in)
                all_margin.append(margin_in)
                #obj_fun.append(obj_in)


                counter+=1








            k+=1

            #print(eslack2)
            C_minus=min(2*C_minus,self.Cu)
            C_plus=min(2*C_plus,self.Cu)
            print("Loop "+str(k)+" C_Unlabeled="+str(self.Cu)+"   Cplus="+str(C_plus)+"   Cminus="+str(C_minus))

            for i in range(len(Y2)):
                if (Y2[i] == 1):
                    sample_weight[len(X1)+i] = C_plus
                else:
                    sample_weight[len(X1)+i] = C_minus

        self.Yresult=Y2
        Y3 = np.concatenate((Y1, Y2), axis=0)
        Y3=Y3.reshape(-1)
        end=time.time()
        print("The training finish in  "+str(end-t)+"  seconds")
        return self,f1_iter,all_margin








    def checkCondition(self,Y,slack):
        '''
        Check condition of the loop 2
        Parameters
        ----------

        Y: labels of X2
                np.array, shape:[n1, ], n1: numbers of samples with semi-labels

        slack: slack variable for unlabeled set
                np.array, shape:[n1, ], n1: numbers of with semi-labels


        '''
        condition=False
        M=len(Y)
        for i in range(M):
            for j in range(M):
                #print('yi='+str(Y[i])+'yj'+str(Y[j])+'slack i='+str(slack[i])+'slack j='+str(slack[j]))
                if((Y[i]!=Y[j]) and (slack[i]>0) and (slack[j]>0) and ((slack[i]+slack[j])>2.001)  ):
                    zeta_total.append(slack[i]+slack[j])
                    condition=True
                    return condition
        return condition

    def getIndexCondition(self,Y,slack):
        '''
        Get index that satisfies condition of loop 2
        Parameters
        ----------

        Y: labels of X2
                np.array, shape:[n1, ], n1: numbers of samples with semi-labels

        slack: slack variable for unlabeled set
                np.array, shape:[n1, ], n1: numbers of with semi-labels


        '''
        M=len(Y)
        for i in range(M):
            for j in range(M):
                if(Y[i]!=Y[j] and slack[i]>0 and slack[j]>0 and ((slack[i]+slack[j])>2.001) ):
                    return i,j

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self):
        return self.clf.predict_proba()

    def decision_function(self, X):
        return self.clf.decision_function(X)
    def sv(self):
        s_v=self.clf.support_vectors_
        return s_v

    def getResult(self):
        return self.Yresult

    def pr(self):
        z=X3
        l=Y3
        #new=np.append(z,l,axis=1)
        return z,l

