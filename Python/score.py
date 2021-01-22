def score(misclassification, accuracy, distance):
    return 0.4 * accuracy + 0.4 * (1 - misclassification) + 0.2 * (1-(distance/100))


m1 = 0.133
a1 = 0.75
d1 = 15
score1 = score(m1, a1, d1)
print("1:", score1)

m2 = 0.05
a2 = 0.7833
d2 = 14
score2 = score(m2, a2, d2)
print("2:", score2)

m3 = 0.1458
a3 = 0.7292
d3 = 15
score3 = score(m3, a3, d3)
print("3:", score3)

m4 = 0.1
a4 = 0.8167
d4 = 14
score4 = score(m4, a4, d4)
print("4:", score4)

m5 = 0.267
a5 = 0.4881
d5 = 7
score5 = score(m5, a5, d5)
print("5:", score5)

m_overall = 0.0760
a_overall = 0.7231
d_overall = 13
score_overall = score(m_overall, a_overall, d_overall)
print("overall:", score_overall)