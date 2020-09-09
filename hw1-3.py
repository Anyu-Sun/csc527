inputlist=[]
sum=0

while True:
    signal_number = int(input("Please enter the number of input signal"))
    if 1 < signal_number:
        break
    else:
        print("Wrong input, please try again")

for i in range(0,signal_number):
    while True:
        input_signal = int(input("Please enter the input signal"))
        if 0 <= input_signal and input_signal <= 1:
            break
        else:
            print("Wrong input, please try again")
    inputlist.append(input_signal)

a = 0-signal_number
b = a-1
bias=[a,-1,0,b]

for i in inputlist:
    sum+=int(i)


AND_gate = (sum+bias[0])
OR_gate = (sum+bias[1])
Firing_state = (sum+bias[2])
Quiescent_state = (sum+bias[3])

def model(status):
    if status >= 0:
        output_status = 1
    elif status <0:
        output_status =0
    return output_status

output_AND = model(AND_gate)
output_OR = model(OR_gate)
output_Firing = model(Firing_state)
output_Quiescent = model(Quiescent_state)

print("Your input signals are:",end='')
for i in inputlist:
    print(i,end='')
print("")
print("AND gate:",output_AND)
print("OR gate:",output_OR)
print("Firing_state:",output_Firing)
print("Quiescent_state:",output_Quiescent)
