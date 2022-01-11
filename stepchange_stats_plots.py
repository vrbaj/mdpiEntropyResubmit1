import matplotlib.pyplot as plt

avg_snr_norm = []
gpd_norm = []
elbnd_norm = []
le_norm = []
e_norm = []

with open("normal_01.csv") as f_normal:
    for line in f_normal:
        split_line = line.split(",")
        avg_snr_norm.append(float(split_line[1]))
        gpd_norm.append(float(split_line[2]))
        elbnd_norm.append(float(split_line[3]))
        le_norm.append(float(split_line[4]))
        e_norm.append(float(split_line[5]))



fig = plt.figure(11, figsize=(6.5, 4.875))
plt.plot(avg_snr_norm, gpd_norm, label="ESE")
plt.plot(avg_snr_norm, elbnd_norm, label="ELBND", linestyle="dashed")
plt.plot(avg_snr_norm, le_norm, label="LE", linestyle="-.")
plt.plot(avg_snr_norm, e_norm, label="ERR", linestyle=":" )
plt.xlim([0, 45])
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("$Úspěšnost$ $detekce$ [-]")
plt.grid()
plt.ylim([.0, 1.01])
plt.tight_layout()
plt.savefig('stepnorm_stats.eps', format='eps', dpi=300)



avg_snr_uniform = []
gpd_uniform = []
elbnd_uniform = []
le_uniform = []
e_uniform = []

with open("uniform.csv") as f_uniform:
    for line in f_uniform:
        split_line = line.split(",")
        avg_snr_uniform.append(float(split_line[1]))
        gpd_uniform.append(float(split_line[2]))
        elbnd_uniform.append(float(split_line[3]))
        le_uniform.append(float(split_line[4]))
        e_uniform.append(float(split_line[5]))


fig = plt.figure(2, figsize=(6.5, 4.875))
plt.plot(avg_snr_uniform, gpd_uniform, label="ESE")
plt.plot(avg_snr_uniform, elbnd_uniform, label="ELBND", linestyle="dashed")
plt.plot(avg_snr_uniform, le_uniform, label="LE", linestyle="-.")
plt.plot(avg_snr_uniform, e_uniform, label="ERR", linestyle=":" )
plt.xlim([0, 40])
plt.legend()
plt.xlabel("SNR [dB]")
plt.ylabel("$Úspěšnost$ $detekce$ [-]")
plt.grid()
plt.ylim([.00, 1.01])
plt.tight_layout()
plt.savefig('stepuni_stats.eps', format='eps', dpi=300)
plt.show()