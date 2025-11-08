
import matplotlib.pyplot as plt


def ploat(f1_iter_seq,all_margin):
  counter=[]
  for i in range(len(f1_iter_seq)):
    counter.append(i)

  plt.figure(figsize=(15,5))
  plt.plot(counter, f1_iter_seq)

  plt.title('F1 over iterations')
  plt.xlabel('iterations')
  plt.ylabel('F1')
  plt.show()

  plt.figure(figsize=(15,5))
  plt.plot(counter, all_margin)

  plt.title('margin size over iterations')
  plt.xlabel('iterations')
  plt.ylabel('margin size')
  plt.show()

  # plt.figure(figsize=(15,5))
  # plt.plot(counter, obj_fun)

  # plt.title('obj function over iterations')
  # plt.xlabel('iterations')
  # plt.ylabel('objective function')
  # plt.show()

