
import pandas as pd
import matplotlib.pyplot as plt
 
days = 365
 
people_list = []
probability_list = []
 
for people in range(1, 31):
 
    probability_different = 1
 
    for i in range(people):
        probability_different *= (days - i) / days
 
    probability_same = 1 - probability_different
 
    people_list.append(people)
    probability_list.append(probability_same)

df = pd.DataFrame({
    "People": people_list,
    "Probability": probability_list
})
 
print(df)
 
plt.figure(figsize=(10, 7))
 
plt.plot(
    df["People"],
    df["Probability"],
    marker="o",
    color="pink",
    linewidth=2
)
 
plt.xlabel("Number of People")
plt.ylabel("Probability of Same Birthday")
plt.title("Birthday Problem")
 
plt.grid(True)
plt.xticks(range(1, 31))
 
plt.show()
plt.show
 