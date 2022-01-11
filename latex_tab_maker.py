avg_snr = []
gpd = []
elbnd = []
le = []
e = []
data_lines = 0

with open("trend.csv") as f_normal:
    for line in f_normal:
        split_line = line.split(",")
        avg_snr.append(float(split_line[1]))
        gpd.append(100 * float(split_line[2]))
        elbnd.append(100 * float(split_line[3]))
        le.append(100 * float(split_line[4]))
        e.append(100 * float(split_line[5]))
        data_lines += 1

with open("latex_tab.txt", "w") as latex:
    latex.write("\\begin{table}\n")
    latex.write("\\begin{center}\n")
    latex.write("\\begin{tabular}{|c|c|c|c|c|}\n")
    latex.write("\\hline\n")
    latex.write("\\textbf{SNR} & \\textbf{ESE [\%]} & \\textbf{ELBND [\%]} & \\textbf{LE [\%]} & \\textbf{Err [\%]} \\\ \n")
    latex.write("\\hline\n")
    for i in range(data_lines):
        print(i)
        latex.write("{0:.2f} & {1:.1f} & {2:.1f} & {3:.1f} & {4:.1f} \\\ \n".format(avg_snr[i],
                                                                         gpd[i], elbnd[i], le[i], e[i]))
    latex.write("\\hline\n")
    latex.write("\\end{tabular}\n")
    latex.write("\\end{center}\n")
    latex.write("\\end{table}")
