from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import datetime as dt


file_selected = False;
file_to_visualize = "";

# loop to break when user selects valid file to view
while not file_selected:

    user_input = input("Enter stock to visualize (ex: \"TD\"), or \"ls\" to see options: ")

    # print all stock .csv files in directory
    if user_input == "ls":
        for file in Path("stock_data/").iterdir():
                print(file.name)

    # break only if input matches filename in directory
    else:
        for file in Path("stock_data/").iterdir():
                # match stock file format to user input
                if (user_input.upper() + ".csv" == file.name):
                    file_to_visualize = user_input.upper()
                    file_selected = True;


"""
Display requested stock data
"""

stock_data = pd.read_csv("stock_data/" + file_to_visualize + ".csv", index_col="Date")





plt.figure(figsize=(12, 8))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=180))
x_dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in stock_data.index.values]

plt.plot(x_dates, stock_data["High"], label="High")
plt.plot(x_dates, stock_data["Low"], label="Low")
plt.xlabel("Time")
plt.ylabel("CAD")
plt.title(file_to_visualize)
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()
