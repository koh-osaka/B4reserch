#coding:utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy
from fractions import Fraction
import datetime
from decimal import Decimal, getcontext

#保存用フォルダ作成
import os
dirname= ''             #プログラムを置いたファイル名
new_dir_path = ('C:/python/experiment/%s/out' %dirname)
os.makedirs(new_dir_path, exist_ok=True)



#ファイルから条件値を読み込む
with open('initial_condition6' , mode = 'rt' , encoding = 'utf-8') as f:
    data_list = f.read()
data = data_list.splitlines()

#初期値を決定
s   = int(data[1])         #衝突回数
Na0 = int(data[3])             #超粒子数
Na1 = int(data[5])
Nb  = int(data[7])
ma0 = 1.67*(10**(-27))    #質量(イオン化した水素原子を仮定)
ma1 = 1.67*(10**(-27))   #-27
mb  = 1.67*(10**(-29))    #電子を仮定
wa0 = 1*(10**10)                 #重み
wa1 = int(data[17])*wa0         #data[17]は重みの比
wb  = int(data[15])*wa0          #data[15]は重みの比
TA0 = int(data[9])                #衝突前の温度[eV]
TA1 = int(data[11])
TB =  int(data[13])

Na = Na0 + Na1
Na_list = [Na0,Na1]
ma_list = [ma0,ma1]
mr0 = ma0*mb/(ma0+mb)      #相対m質量
mr1 = ma1*mb/(ma1+mb)
mr_list = [mr0,mr1]
wa_list = [wa0,wa1]
TA_list = [TA0,TA1] 
 
kb = 1.38*(10**-23)     #ボルツマン定数
eV_T = 1.16045*(10**4)
Ea0 = 4*1.6*(10**-19)    #粒子の電荷
dt = 1*(10**(-10))       #時間の刻み幅
Ea1 = 4*1.6*(10**-19)
Ea_list = [Ea0,Ea1]
Eb = -1*1.6*(10**-19)
N =  10**20             #本来はNa,Nbのうち小さいほうの密度
ram = 15                #λ:クーロン対数(今は仮)
ip = 8.85*(10**(-12))   #ε0:真空の誘電率

Nf = Fraction(Na,Nb)    #超粒子比(分数)
Ni = int(Nf)            #比の整数部
Nr = Nf - Ni            #比の小数部


#ベクトルの絶対値を返す関数
def f_norm(v):
    V=(v[0]**2 + v[1]**2 + v[2]**2)**0.5
    return V


#更新速度duの各成分を求める
def f1(ux,uy,uz,uu,numA):             
    
    uv = (ux**2 + uy**2)**0.5   #uvはxy平面での速度。vは垂直の意味
    phi1 = 2*np.pi*np.random.random()
    sinp1 = np.sin(phi1)
    cosp1 = np.cos(phi1)
    
    #theを論文p15.(3.7a)から計算
    sigma = ((dt*(Ea_list[numA]**2)*(Eb**2)*N*ram)/(8*np.pi*(ip**2)\
              *(mr_list[numA]**2)*(uu**3)))**0.5 #sigma:theの標準偏差
    
    if sigma**2 <= 1 :
        the = np.random.normal(loc = 0,scale = sigma,size = 1)   #ガウス分布から決定    
        sint1 = (2*the)/(1+(the**2))
        cost1 = 1 - 2*(the**2)/(1+(the**2))
        C1 = 1 - cost1         #C=1-cos とおいた  
    else:
        the = np.pi*np.random.uniform(0,1)      
        sint1 = np.sin(the)
        cost1 = np.cos(the)
        C1 = 1 - cost1
    phi2 = phi1 + 2*np.pi*(1-(wa_list[numA]/wb)**0.5)*(1 - np.random.uniform(0,1))
    sinp2 = np.sin(phi2)
    cosp2 = np.cos(phi2)
    C2 = (wa_list[numA]/wb)*(1-cost1)
    if sint1 >= 0 :
        x = 1
    else:
        x = -1
    sint2 = x*(((wa_list[numA]/wb)*(sint1**2)+(wa_list[numA]/wb)\
                *(1-(wa_list[numA]/wb))*((1-cost1)**2))**0.5)
    
    dux1 = float((ux/uv)*uz*sint1*cosp1 - (uy/uv)*uu*sint1*sinp1 - ux*C1)
    duy1 = float((uy/uv)*uz*sint1*cosp1 + (ux/uv)*uu*sint1*sinp1 - uy*C1)
    duz1 = float(- uv*sint1*cosp1 - uz*C1)
    du1 = np.array([dux1,duy1,duz1])
    dux2 = float((ux/uv)*uz*sint2*cosp2 - (uy/uv)*uu*sint2*sinp2 - ux*C2)
    duy2 = float((uy/uv)*uz*sint2*cosp2 + (ux/uv)*uu*sint2*sinp2 - uy*C2)
    duz2 = float(- uv*sint2*cosp2 - uz*C2)
    du2 = np.array([dux2,duy2,duz2])
    return du1 , du2 
    


#補正速度のパラメータを求める    
def f2(): 
    Va1 = []
    Vb1 = []
    for k in range(Na):
        Va1.append(dict_A[k]["v"])
    for k in range(Nb):
        Vb1.append(dict_B[k]["v"])
    Va2 = []
    Vb2 = []
    for k in range(Na):
        Va2.append(dict_A[k]["v!"])
    for k in range(Nb):
        Vb2.append(dict_B[k]["v!"])
    Van1 = []
    Vbn1 = []
    for k in range(Na):
        Van1.append(np.linalg.norm((dict_A[k]["v"])**2))
    for k in range(Nb):
        Vbn1.append(np.linalg.norm((dict_B[k]["v"])**2))
    Van2 = []
    Vbn2 = []
    for k in range(Na):
        Van2.append(f_norm((dict_A[k]["v!"])**2))
    for k in range(Nb):
        Vbn2.append(f_norm((dict_B[k]["v!"])**2))
    M =  wa0*ma0*Na0 + wa1*ma1*Na1 + wb*mb*Nb
    V = (wa0*ma0*np.sum(np.array(Va1)[:Na0],axis=0) \
         + wa1*ma1*np.sum(np.array(Va1)[Na0:], axis=0) \
         + wb*mb*np.sum(np.array(Vb1), axis=0))/M
    dV = (wa0*ma0*np.sum(np.array(Va2)[:Na0], axis=0) \
         + wa1*ma1*np.sum(np.array(Va2)[Na0:], axis=0) \
         + wb*mb*np.sum(np.array(Vb2), axis=0))/M \
             -V
    E = 0.5*wa0*ma0*sum(Van1[:Na0]) + 0.5*wa1*ma1*sum(Van1[Na0:]) + 0.5*wb*mb*sum(Vbn1)
    
    dE = (0.5*wa0*ma0*sum(Van2[:Na0]) + 0.5*wa1*ma1*sum(Van2[Na0:]) + 0.5*wb*mb*sum(Vbn2)) \
        - E
    alf = ((E - 0.5*M*f_norm(V**2))/(E + dE - 0.5*M*f_norm((V + dV)**2)))**0.5
    return V, alf, dV, dE, E

#衝突自体の関数 
def f3(na,nb,numA,num3):
     va = dict_A[na-1]["v!"] 
     vb = dict_B[nb-1]["v!"]    
     u = va - vb                             #相対速度
     uu = np.linalg.norm(u)                  #相対速度の大きさ
     ux = u[0]
     uy = u[1]
     uz = u[2]
     result1 = f1(ux,uy,uz,uu,numA)          #衝突
     vaa = va + (mr_list[numA]/ma_list[numA])*result1[0] 
     vbb = vb - (mr_list[numA]/mb)*result1[1]
     return vaa,vbb

        
#全運動エネルギーを計算する関数           
def f_UE():
    UU2 = list()
    for i in range(2):
        for k in range(Na_list[i]):
            UU2.append(0.5*ma_list[i]*wa_list[i]*\
                       ((np.linalg.norm(dict_A[i*Na_list[0]+k]["v!"]))**2))
    for k in range(Nb):
        UU2.append(0.5*mb*wb*((np.linalg.norm(dict_B[k]["v!"]))**2))
    return sum(UU2)

#全運動量を計算する関数
def f_UR():
    UU3 = list()    
    for i in range(2):
        for k in range(Na_list[i]):
            UU3.append(ma_list[i]*wa_list[i]*dict_A[i*Na_list[0]+k]["v!"])   

    for k in range(Nb):
        UU3.append(mb*wb*dict_B[k]["v!"])
    return np.linalg.norm(np.sum(UU3,axis=0))

#衝突前の速度をヒストグラムに表すためにリストに保存           
def f_hist(A0,A1,B):
    VNn_list=[]
    for i in range(3):
        Vn_list=[]
        for k in range(A0*Na0 + A1*Na1):
            Vn_list.append(dict_A[(1-A0)*Na0 + k]["v!"][i])
        for k in range(B*Nb):
            Vn_list.append(dict_B[k]["v!"][i])
        VNn_list.append(Vn_list)
    return VNn_list
 
#衝突前後の速度をヒストグラムに表示        
def f_hist2(A0,A1,B):
    d=["X","Y","Z"]
    N=["A0","A1","B","ALL"]
   
    if A0==1 and A1==0 and B==0:
        nu=0
    if A0==0 and A1==1 and B==0:
        nu=1
    if A0==0 and A1==0 and B==1:
        nu=2
    if A0==1 and A1==1 and B==1:
        nu=3
    for i in range(3):
        Vnn_list=[]
        for k in range(A0*Na0 + A1*Na1):
            Vnn_list.append(dict_A[(1-A0)*Na0 + k]["v!"][i])
        for k in range(B*Nb):
            Vnn_list.append(dict_B[k]["v!"][i])
        binxx = 20               #衝突後のヒストグラムのbins数
        width = (max(Vnn_list) - min(Vnn_list))/binxx    #階級幅を決定
        binx =  int((max(VN_list[nu][i]) - min(VN_list[nu][i]))/width)     
        if binx<=1:
            binx=3
        plt.figure()
        plt.hist(VN_list[nu][i],label="before",alpha=0.5,color='b',bins=binx,histtype="stepfilled")
        plt.hist(Vnn_list,label="after",alpha=0.5,color='r',bins=binxx,histtype="stepfilled")
        plt.legend(['before','after'])
        plt.title("velocity in %s-direction" %d[i])
        plt.xlabel("velocity[m/s]")
        plt.ylabel("volume")
        plt.savefig('%s/figSYM4-%s-%s.png' %(new_dir_path,N[nu],d[i]))
        
def f_temp():
    Ta0 = [0,0,0]          #a0の各成分の温度
    for i in range(3):
        U4 = list()
        U5 = list()
        for k in range(Na0):
            U4.append((dict_A[k]["v!"][i])**2)
    
        for k in range(Na0):
            U5.append(dict_A[k]["v!"][i])
        Ta0[i] = ma0*(np.sum(U4)/Na0-((np.sum(U5)/Na0)**2))/(kb*eV_T)
    Ta1 = [0,0,0]          #a1の各成分の温度
    for i in range(3):
        U4 = list()
        U5 = list()
        for k in range(Na1):
            U4.append((dict_A[Na0+k]["v!"][i])**2)
    
        for k in range(Na1):
            U5.append(dict_A[Na0+k]["v!"][i])
        Ta1[i] = ma1*(np.sum(U4)/Na1-((np.sum(U5)/Na1)**2))/(kb*eV_T)
    Tb = [0,0,0]          #bの各成分の温度
    for i in range(3):
        U4 = list()
        U5 = list()
        for k in range(Nb):
            U4.append((dict_B[k]["v!"][i])**2)
        for k in range(Nb):
            U5.append(dict_B[k]["v!"][i])
        Tb[i] = mb*(np.sum(U4)/Nb-((np.sum(U5)/Nb)**2))/(kb*eV_T)
    return sum(Ta0),sum(Ta1),sum(Tb)

#ディクショナリの箱を作成
dict_A = []
for i in range(Na0):
    dict_a = {'n':i, 'v':np.array([0,0,0]) , 'v!':np.array([0,0,0])}
    dict_A.append(dict_a)

for i in range(Na1):
    dict_a = {'n':Na0+i,'v':np.array([0,0,0]) , 'v!':np.array([0,0,0])}
    dict_A.append(dict_a)
    
dict_B = []
for i in range(Nb):
    dict_b = {'n':i,'v':np.array([0,0,0]) , 'v!':np.array([0,0,0])}
    dict_B.append(dict_b)
    
    
#用意した箱にランダムな速度ベクトルを代入(温度から決定)
for i in range(2):
    for j in range(Na_list[i]):
        sig = ((kb*TA_list[i]*eV_T)/(ma_list[i]*3))**0.5             #xyz各方向の温度なので3で割る
        dict_A[i*Na_list[0]+j]["v!"] = np.array([np.random.normal(loc=0,scale=sig),\
                                                np.random.normal(loc=0,scale=sig),np.random.normal(loc=0,scale=sig)])
   
for j in range(Nb):
    sig = ((kb*TB*eV_T)/(mb*3))**0.5
    dict_B[j]["v!"] = np.array([np.random.normal(loc=0,scale=sig),\
                               np.random.normal(loc=0,scale=sig),np.random.normal(loc=0,scale=sig)])

#衝突前の速度分布を保存
VN_list=[]
VN_list.append(f_hist(A0=1,A1=0,B=0))
VN_list.append(f_hist(A0=0,A1=1,B=0))
VN_list.append(f_hist(A0=0,A1=0,B=1))
VN_list.append(f_hist(A0=1,A1=1,B=1))


UE=[]          #運動エネルギーの変化をプロットするためのリスト
UR=[]         #運動量の変化をプロットするためのリスト
Ta0_list=[]
Ta1_list=[]
Tb_list=[]
dE_list=[]      #dE/Eをステップごとに保存
num4 = 0
#衝突のループ(num1:衝突回数)
num1 = 1
while num1 <= s:
    #この時点での全運動エネルギーをリストに保存
    UE1 = f_UE()
    UE.append(UE1)

    #この時点での全運動量をリストに保存
    UR1 = f_UR()
    UR.append(UR1)
    
    #この時点での電子温度をリストに保存
    temp = f_temp()
    Ta0_list.append(temp[0])
    Ta1_list.append(temp[1])
    Tb_list.append(temp[2])
    
    #衝突前の速度を粒子毎に保存
    for i in range(Na):
        dict_A[i]["v"] = dict_A[i]["v!"]
    for i in range(Nb):
        dict_B[i]["v"] = dict_B[i]["v!"]
    
    #group1
    num2 = 1    #num2:粒子bの番号           
    while num2 <= Nr*Nb :
        
        #ペアリング(num3:ペアを組む回数)
        num3= 1
        while num3 <= Ni + 1:
            if Na>Nb:
                na = (num2-1)*(Ni+1)+num3
                nb = num2
            else:               #Na<Nbの場合はペアリング逆
                na = num2
                nb = (num2-1)*(Ni+1)+num3
            #粒子Aの中の種類を指定
            if na<=Na0:
                numA = 0
            else:
                numA = 1
                        
            result3 = f3(na,nb,numA,num3)
            va = result3[0]
            vb = result3[1]
        
            #ディクショナリディクショナリ書き換え
            dict_A[na-1]["v!"] = np.array(va)
            dict_B[nb-1]["v!"] = np.array(vb)
            num3 += 1
        num2 += 1
    #group2
    while num2 <= Nb :
        #ペアリング(num3:ペアを組む回数)
        num3= 1
        while num3 <= Ni :
            if Na>Nb:
                na = int((Ni + 1)*Nr*Nb + (num2-Nr*Nb-1)*Ni + num3) 
                nb = num2
            else:
                na = num2
                nb = int((Ni + 1)*Nr*Nb + (num2-Nr*Nb-1)*Ni + num3)
                
            if na<=Na0:
                numA = 0
            else:
                numA = 1
           
            result3 = f3(na,nb,numA,num3)
            va = result3[0]
            vb = result3[1]
            
            #ディクショナリディクショナリ書き換え
            dict_A[na-1]["v!"] = np.array(va)
            dict_B[nb-1]["v!"] = np.array(vb)
            num3 += 1
        num2 += 1
    
    #速度補正を実行
    Col =f2()
    V   = Col[0]
    alf = Col[1]
    dV  = Col[2]
    dE  = Col[3]
    E   = Col[4]
    dE_list.append(dE/E)
    
    for k in range(Na):
        v = dict_A[k]["v!"] 
        vv = V + alf*(v - V - dV)
        dict_A[k]["v!"] = vv
    for k in range(Nb):
        v = dict_B[k]["v!"] 
        vv = V + alf*(v - V - dV)
        dict_B[k]["v!"] = vv
        
    dict_A0 = dict_A[:Na0]
    dict_A1 = dict_A[Na0:]
    np.random.shuffle(dict_A0)                   #各ディクショナリのシャッフル(A0)
    np.random.shuffle(dict_A1)                   #(A1)
    dict_A = dict_A0 + dict_A1
       
    np.random.shuffle(dict_B)                         #(B)
    num1 += 1
    
    #衝突状況が10分のいくらか表示
    if s>=500:
        for i in range(1,10):
            if num1 == (i)*(s/10):
                print("%s /10    time = %s " %(i,datetime.datetime.now()))
    


#衝突前後の変化を表示
print("運動エネルギー(衝突前) = %s" %UE[0])
print("運動エネルギー(衝突後) = %s" %UE[-1])
print("運動量(衝突前) = %s" %UR[0])
print("運動量(衝突後) = %s" %UR[-1])
print("Ta0(衝突前)=%s" %Ta0_list[0])
print("Ta0(衝突後)=%s" %Ta0_list[-1])
print("Ta1(衝突前)=%s" %Ta1_list[0])
print("Ta1(衝突後)=%s" %Ta1_list[-1])
print("Tb(衝突前)=%s" %Tb_list[0])
print("Tb(衝突後)=%s " %Tb_list[-1])
print("dE/E=%s " %np.mean(np.abs(dE_list)))

#電子温度の変化をグラフに出力
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("T(eV)")
ax.set_xlabel("number of time iteration")
ax.plot(Ta0_list,label="Ta0",color='b')
ax.plot(Ta1_list,label="Ta1",color='r')
ax.plot(Tb_list,label="Tb",color='g')
ax.tick_params(direction='in' , bottom=True ,top=True ,left=True ,right=True)
plt.ylim(0,)
plt.xlim(0,)
plt.legend(['Ta0','Ta1','Tb'])
plt.savefig('%s/figSYM4-T.png' %new_dir_path)

#衝突後の速度分布ををヒストグラムで表示
f_hist2(A0=1,A1=0,B=0)
f_hist2(A0=0,A1=1,B=0)
f_hist2(A0=0,A1=0,B=1)
f_hist2(A0=1,A1=1,B=1)

#運動量と運動エネルギーの変化をグラフ表示
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("UE/UE0")
ax.set_xlabel("number of time iteration")
ax.tick_params(direction='in' , bottom=True ,top=True ,left=True ,right=True)
ax.plot(UE/UE[0])
plt.ylim(0,2)
plt.xlim(0,)
plt.savefig('%s/figSYM4-UE.png' %new_dir_path)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("UR/UR0")
ax.set_xlabel("number of time iteration")
ax.tick_params(direction='in' , bottom=True ,top=True ,left=True ,right=True)
ax.plot(UR/UR[0])
plt.ylim(0,2)
plt.xlim(0,)
plt.savefig('%s/figSYM4-UR.png' %new_dir_path)

#dE/Eの散布図をグラフに出力
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("dE/E")
ax.set_ylabel("dE/E[%]")
ax.set_xlabel("number of time iteration")
ax.tick_params(direction='in' , bottom=True ,top=True ,left=True ,right=True)
ax.plot(dE_list ,marker=".",linestyle='None')
plt.savefig('%s/figSYM4-dE.png' %new_dir_path)

#UE誤差平均を計算
UEerr=0
for i in range(s):
    UEerr += abs(UE[i] - UE[0])/s
    
#UR誤差平均を計算
URerr=0
for i in range(s):
    URerr += abs(UR[i] - UR[0])/s
    
#結果をファイルに出力
path = ('%s/result.txt' %new_dir_path)
with open(path,mode='w') as f:
    f.write('s =%s\n' %s)
    f.write('Na0 =%s\n' %Na0)
    f.write('Na1 =%s\n' %Na1)
    f.write('Nb =%s\n' %Nb)
    f.write('ma0 =%s\n' %ma0)
    f.write('ma1 =%s\n' %ma1)
    f.write('mb =%s\n' %mb)
    f.write('wa0 =%s\n' %wa0)
    f.write('wb =%s\n' %wb)
    f.write('TA0 =%s\n' %TA0)
    f.write('TA1 =%s\n' %TA1)
    f.write('Tb =%s\n' %TB)
    f.write('dt =%s\n' %dt)
    f.write('\n')
    f.write("Ta0(衝突前)=%s\n" %Ta0_list[0])
    f.write("Ta0(衝突後)=%s\n" %Ta0_list[-1])
    f.write("Ta1(衝突前)=%s\n" %Ta1_list[0])
    f.write("Ta1(衝突後)=%s\n" %Ta1_list[-1])
    f.write("Tb(衝突前)=%s\n" %Tb_list[0])
    f.write("Tb(衝突後)=%s\n" %Tb_list[-1])
    f.write("運動エネルギー(衝突前) = %s\n" %UE[0])
    f.write("運動エネルギー(衝突後) = %s\n" %UE[-1])
    f.write("運動量(衝突前) = %s\n" %UR[0])
    f.write("運動量(衝突後) = %s\n" %UR[-1])
    f.write("dE/E=%s \n" %np.mean(np.abs(dE_list)))
    f.write("UE誤差平均=%s \n" %UEerr)
    f.write("UR誤差平均=%s \n" %URerr)

path = ('%s/data-T.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(s):
        f.write("%s %s %s\n" %(Ta0_list[i],Ta1_list[i],Tb_list[i]))
path = ('%s/data-UE.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(s):
        f.write("%s\n" %UE[i])
path = ('%s/data-UR.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(s):
        f.write("%s\n" %UR[i])
path = ('%s/data-Va0.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(Na0):
        f.write("%s\n" %dict_A[i]["v!"])
path = ('%s/data-Va1.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(Na1):
        f.write("%s\n" %dict_A[Na0+i]["v!"])
path = ('%s/data-Vb.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(Nb):
        f.write("%s\n" %dict_B[i]["v!"])
path = ('%s/data-dE.txt' %new_dir_path)
with open(path,mode='w') as f:
    for i in range(s):
        f.write("%s\n" %dE_list[i])


print('<計算終了>')
