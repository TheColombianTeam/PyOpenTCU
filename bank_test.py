from TensorBuffer import Bank

bank = Bank(data_width=32, memory_size=512)
bank.write('33', '0cacqwe20cacqwe2')
bank.write('00', '0ca1aczz0ca1aczz')
bank.write('01', '0c1c34550c1c3455')
bank.write('22', '03ac123403ac1234')
bank.write('23', '0c4598760c459876')

print('Buffer\n{}'.format(bank))