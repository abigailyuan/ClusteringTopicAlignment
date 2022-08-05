import matplotlib.pyplot as plt
from scipy.stats import pearsonr

F = [190,100,350,200,300]
M = [220,200,50,60,600]
ShopNum = [40,30,30,25,90]
P = []
for i in range(5):
    P.append(F[i]+M[i])

# dataset = sorted(list(zip(P, ShopNum)))
# x = [x[0] for x in dataset]
# y = [x[1] for x in dataset]
# plt.plot(x, y)
# plt.xlabel('Population')
# plt.ylabel('Number of Shops')
# plt.title('Population vs. Number of shops')
# # plt.show()
# plt.savefig('Population.png')

p = pearsonr(P, ShopNum)
print(p)