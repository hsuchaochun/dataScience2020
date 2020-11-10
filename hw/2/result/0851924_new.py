
net = AttractivenessRNN(weights, weights.shape[1], 8, 2)
net.load_state_dict(torch.load('./weights/weights.pth'))
if (train_on_gpu):
        net.cuda()
else:
    net.cpu()
    
net.eval()

pred_input = test_data_only_df['word_list'].values.tolist()
pred_input = torch.LongTensor(pred_input)
pred_input.size()

h = net.init_hidden(227)
h = tuple([each.data for each in h])
pred, h = net(pred_input.cuda(),h)

pred

pred_output = pred.data.tolist()

# output prediction

new_df=pd.DataFrame()
new_df['Label'] = pred_output
new_df.index = np.arange(1, len(new_df) + 1)
new_df.to_csv('ans_shit.csv', index_label='ID')
