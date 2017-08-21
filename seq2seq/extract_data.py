import os

def main():
  fname = "perplexity_data.txt"
  content = []
  with open(fname) as f:
    content = f.readlines()

  steps, lr, st, perpl = [], [], [], []
  for line in content:
    if "global step" in line:
      line_arr = line.split(" ")
      steps.append(int(line_arr[2]))
      lr.append(float(line_arr[5]))
      st.append(float(line_arr[7]))
      perpl.append(float(line_arr[9]))

  # for i in range(len(steps)):
    # print lr[i]
  # for i in range(len(steps)):
    # print st[i]
  for i in range(len(steps)):
    print perpl[i]

main()

