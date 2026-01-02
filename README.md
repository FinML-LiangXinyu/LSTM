# 长短期记忆网络

## 一、 $LSTM$ 算法原理

对于长度为 $T$ 的时间序列数据 $x=\left[x_1,x_2,\ldots,x_t,\ldots,x_T\right]$ ， $x_t$ 为时刻 $t$ 的输入向量。$LSTM$ 算法的结构单元如下：

$i_t=\sigma\left(z_{i_t}\right)=\sigma\left(U_ih_{t-1}+W_ix_t+b_i\right)$

$f_t=\sigma\left(z_{f_t}\right)=\sigma\left(U_fh_{t-1}+W_fx_t+b_f\right)$

$o_t=\sigma\left(z_{o_t}\right)=\sigma\left(U_oh_{t-1}+W_ox_t+b_o\right)$

$\widetilde{c_t}=tanh\left(U_ch_{t-1}+W_cx_t+b_c\right)$

$c_t=i_t\odot\widetilde{c_t}+f_t\odot c_{t-1}$

$h_t=o_t\odot tanh\left(c_t\right)$

$z_t=Vh_t+b_z$

$\widehat{y_t}=softmax\left(z_t\right)$

其中， $h_{t-1}$ 代表 $t-1$ 时刻的隐状态， $x_t$ 为时刻 $t$ 的输入， $i_t$ 为时刻 $t$ 的输入门， $f_t$ 为时刻 $t$ 的遗忘门， $o_t$ 为时刻 $t$ 的输出门， $c_t$ 为时刻 $t$ 的细胞状态，它由经过输入门 $i_t$ 控制的 $\widetilde{c_t}$ 和遗忘门 $f_t$ 控制的 $c_{t-1}$ 相加得到。时刻 $t$ 的细胞状态 $c_t$ 经过 $tanh(·)$ 激活后，在输出门 $o_t$ 的控制下转换为 $t$ 时刻的隐状态 $h_t$ ，时刻 $t$ 的净输入 $z_t$ 经过 $softmax(·)$ 转换为时刻 $t$ 的最终输出 $\widehat{y_t}$ 。 $U_i$ 、$U_f$ 、 $U_o$ 、 $U_c$ 、 $W_i$ 、 $W_f$ 、 $W_o$ 、 $W_c$ 、 $V$ 为神经网络的权重矩阵， $b_i$ 、 $b_f$ 、 $b_o$ 、 $b_c$ 、 $b_z$ 为神经网络的偏置向量。

矩阵 $U_i$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径：

$U_i\rightarrow i_1,i_2,\ldots,i_t,\ldots,i_T\rightarrow c_1,c_2,\ldots,c_t,\ldots,c_T\rightarrow h_1,h_2,\ldots,h_t,\ldots,h_T\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow L_1,L_2,\ldots,L_t,\ldots,L_T\rightarrow L$

可推：

$dc_t=di_t\odot\widetilde{c_t}={\sigma_{i_t}}^\prime\odot dU_ih_{t-1}\odot\widetilde{c_t}$

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^T{\sigma_{i_t}}^\prime\odot d U_ih_{t-1}\odot\widetilde{c_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\right)^T{\sigma_{i_t}}^\prime\odot d U_i}h_{t-1}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime\right)^TdU_i}h_{t-1}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime{h_{t-1}}^T\right)^TdU_i}\right)$

$\frac{\partial L}{\partial U_i}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime{h_{t-1}}^T}$

同理：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^T{\sigma_{i_t}}^\prime\odot d W_ix_t\odot\widetilde{c_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\right)^T{\sigma_{i_t}}^\prime\odot d W_ix_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime\right)^TdW_ix_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime{x_t}^T\right)^TdW_i}\right)$

$\frac{\partial L}{\partial W_i}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime{x_t}^T}$

$dc_t=di_t\odot\widetilde{c_t}={\sigma_{i_t}}^\prime\odot db_i\odot\widetilde{c_t}$

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^T{\sigma_{i_t}}^\prime\odot d b_i\odot\widetilde{c_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\right)^T{\sigma_{i_t}}^\prime\odot d b_i}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime\right)^Tdb_i}\right)$

$\frac{\partial L}{\partial b_i}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime}$

矩阵 $U_f$ 在每个时刻 $t$ 都会影响该时刻的损失 $L_t$ 进而影响总损失 $L$ ，对应前向链式传播路径：

$U_f\rightarrow f_1,f_2,\ldots,f_t,\ldots,f_T\rightarrow c_1,c_2,\ldots,c_t,\ldots,c_T\rightarrow h_1,h_2,\ldots,h_t,\ldots,h_T\rightarrow z_1,z_2,\ldots,z_t,\ldots,z_T\rightarrow L_1,L_2,\ldots,L_t,\ldots,L_T\rightarrow L$

可推：

$dc_t=df_t\odot c_{t-1}={\sigma_{f_t}}^\prime\odot dU_fh_{t-1}\odot c_{t-1}$

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^T{\sigma_{f_t}}^\prime\odot d U_fh_{t-1}\odot c_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot c_{t-1}\right)^T{\sigma_{f_t}}^\prime\odot d U_fh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot c_{t-1}\odot{\sigma_{f_t}}^\prime\right)^TdU_fh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot c_{t-1}\odot{\sigma_{f_t}}^\prime{h_{t-1}}^T\right)^TdU_f}\right)$

$\frac{\partial L}{\partial U_f}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot c_{t-1}\odot{\sigma_{f_t}}^\prime{h_{t-1}}^T}$

同理：

$\frac{\partial L}{\partial W_f}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot c_{t-1}\odot{\sigma_{f_t}}^\prime{x_t}^T}$

$\frac{\partial L}{\partial b_f}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot c_{t-1}\odot{\sigma_{f_t}}^\prime}$

最后时刻的隐状态 $h_T$ 在时刻 $T$ 会影响该时刻的损失 $L_T$ 进而影响总损失 $L$ ，对应的链式传播路径为：

$h_T\rightarrow z_T\rightarrow L_T\rightarrow L$

可得：

$dL=tr\left(\left(\frac{\partial L}{\partial z_T}\right)^Tdz_T\right)=tr\left(\left(\frac{\partial L}{\partial z_T}\right)^TVdh_T\right)=tr\left(\left(V^T\frac{\partial L}{\partial z_T}\right)^Tdh_T\right)$

$\frac{\partial L}{\partial h_T}=V^T\frac{\partial L}{\partial z_T}=V^T\left(\widehat{y_T}-y_T\right)$

同理可得：

$dL=tr\left(\left(\frac{\partial L}{\partial z_{t-1}}\right)^Tdz_{t-1}\right)=tr\left(\left(\frac{\partial L}{\partial z_{t-1}}\right)^TVdh_{t-1}\right)=tr\left(\left(\frac{\partial L}{\partial z_{t-1}}\right)^TVdh_{t-1}\right)=tr\left(\left(V^T\frac{\partial L}{\partial z_{t-1}}\right)^Tdh_{t-1}\right)$

$\frac{\partial L}{\partial h_{t-1}}=V^T\frac{\partial L}{\partial z_{t-1}}=V^T\left(\widehat{y_{t-1}}-y_{t-1}\right)$

$t$时刻的细胞状态$c_t$通过影响$t$时刻的隐状态$h_t$和$t+1$时刻的隐状态$h_{t+1}$进而影响$t$时刻的损失$L_t$和$t+1$时刻的损失$L_{t+1}$，最终影响总损失$L$，对应的链式传播路径为：

$c_t\rightarrow h_t\rightarrow z_t\rightarrow L_t\rightarrow L$

$c_t\rightarrow c_{t+1}\rightarrow h_{t+1}\rightarrow z_{t+1}\rightarrow L_{t+1}\rightarrow L$

可推：

$dc_{t+1}=f_t\odot dc_t$

$dh_t=o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot dc_t$

$dL=tr\left(\left(\frac{\partial L}{\partial h_t}\right)^Tdh_t\right)+tr\left(\left(\frac{\partial L}{\partial c_{t+1}}\right)^Tdc_{t+1}\right)=tr\left(\left(\frac{\partial L_t}{\partial h_t}\right)^To_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot d c_t\right)+tr\left(\left(\frac{\partial L}{\partial c_{t+1}}\right)^Tf_t\odot d c_t\right)=tr\left(\left(\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^Tdc_t\right)+tr\left(\left(\frac{\partial L}{\partial c_{t+1}}\odot f_t\right)^Tdc_t\right)$

$\frac{\partial L}{\partial c_t}=\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)+\frac{\partial L}{\partial c_{t+1}}\odot f_t$

同理：

$c_T\rightarrow h_T\rightarrow z_T\rightarrow L_T\rightarrow L$

$\frac{\partial L}{\partial c_T}=\frac{\partial L}{\partial h_T}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)$

前一时刻的隐状态 $h_{t-1}$ 会影响时刻 $t$ 的输入门  $i_t$ 、遗忘门 $f_t$ 和 $\widetilde{c_t}$ ，进而影响时刻 $t$ 的细胞状态 $c_t$ ，对应的链式传播路径为：

$h_{t-1}\rightarrow i_t,f_t,\widetilde{c_t}\rightarrow c_t\rightarrow L_t\rightarrow L$

此外，隐状态 $h_{t-1}$ 还有如下两条链式传播路径：

$h_{t-1}\rightarrow z_{t-1}\rightarrow L_{t-1}\rightarrow L$

$h_{t-1}\rightarrow o_t\rightarrow h_t\rightarrow z_t\rightarrow L_t\rightarrow L$

可推：

$dL=tr\left(\left(\frac{\partial L}{\partial z_{t-1}}\right)^Tdz_{t-1}\right)+tr\left(\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t\right)+tr\left(\left(\frac{\partial L}{\partial h_t}\right)^Tdh_t\right)=tr\left(\left(V^T\left(\widehat{y_{t-1}}-y_{t-1}\right)\right)^Tdh_{t-1}\right)+tr\left(\left(\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^Tdc_t\right)+tr\left(\left(\frac{\partial L}{\partial h_t}\right)^Tdo_t\odot t a n h\left(c_t\right)\right)=tr\left(\left(V^T\left(\widehat{y_{t-1}}-y_{t-1}\right)\right)^Tdh_{t-1}\right)+tr\left(\left(\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^Tdc_t\right)+tr\left(\left(\frac{\partial L}{\partial h_t}\right)^T{\sigma_{o_t}}^\prime\odot U_odh_{t-1}\odot t a n h\left(c_t\right)\right)=tr\left(\left(V^T\left(\widehat{y_{t-1}}-y_{t-1}\right)\right)^Tdh_{t-1}\right)+tr\left(\left(\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^Tdc_t\right)+tr\left(\left({U_o}^T\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime\right)^Tdh_{t-1}\right)$

$dc_t=di_t\odot\widetilde{c_t}+i_t\odot d\widetilde{c_t}+df_t\odot c_{t-1}={\sigma_{i_t}}^\prime\odot U_idh_{t-1}\odot\widetilde{c_t}+i_t\odot\left(1-\widetilde{c_t}\odot\widetilde{c_t}\right)\odot U_cdh_{t-1}+{\sigma_{f_t}}^\prime\odot U_fdh_{t-1}\odot c_{t-1}$

$tr\left(\left(\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^Tdc_t\right)=tr\left(\left(\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^T\left({\sigma_{i_t}}^\prime\odot U_idh_{t-1}\odot\widetilde{c_t}+i_t\odot\left(1-\widetilde{c_t}\odot\widetilde{c_t}\right)\odot U_cdh_{t-1}+{\sigma_{f_t}}^\prime\odot U_fdh_{t-1}\odot c_{t-1}\right)\right)=tr\left(\left(\frac{\partial L_t}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^T\left({\sigma_{i_t}}^\prime\odot U_idh_{t-1}\odot\widetilde{c_t}\right)\right)+tr\left(\left(\frac{\partial L_t}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^T\left(i_t\odot\left(1-\widetilde{c_t}\odot\widetilde{c_t}\right)\odot U_cdh_{t-1}\right)\right)+tr\left(\left(\frac{\partial L_t}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\right)^T\left({\sigma_{f_t}}^\prime\odot U_fdh_{t-1}\odot c_{t-1}\right)\right)=tr\left(\left({U_i}^T\frac{\partial L_t}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime\right)^Tdh_{t-1}\right)+tr\left(\left({U_c}^T\frac{\partial L_t}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot i_t\odot\left(1-\widetilde{c_t}\odot\widetilde{c_t}\right)\right)^Tdh_{t-1}\right)+tr\left(\left({U_f}^T\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot c_{t-1}\odot{\sigma_{f_t}}^\prime\right)^Tdh_{t-1}\right)$

可得：

$\frac{\partial L}{\partial h_{t-1}}=V^T\left(\widehat{y_{t-1}}-y_{t-1}\right)+{U_i}^T\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime+{U_c}^T\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot i_t\odot\left(1-\widetilde{c_t}\odot\widetilde{c_t}\right)+{U_f}^T\frac{\partial L}{\partial h_t}\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot c_{t-1}\odot{\sigma_{f_t}}^\prime+{U_o}^T\frac{\partial L}{\partial h_t}\odot tanh\left(c_t\right)\odot{\sigma_{o_t}}^\prime$

时刻 $t$ 的 $\widetilde{c_t}$ 的链式传播路径为：

$\widetilde{c_t}\rightarrow c_t\rightarrow h_t\rightarrow z_t\rightarrow L_t\rightarrow L$

可推：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Tdc_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Ti_t\odot d\widetilde{c_t}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\right)^Ti_t\odot\left(1-{\widetilde{c_t}}^2\right)\odot d U_ch_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right)\right)^TdU_ch_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial c_t}\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right){h_{t-1}}^T\right)^TdU_c}\right)$

可得:

$\frac{\partial L}{\partial U_c}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right){h_{t-1}}^T}$

同理可得：

$\frac{\partial L}{\partial W_c}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right){x_t}^T}$

$\frac{\partial L}{\partial b_c}=\sum_{t=1}^{T}{\frac{\partial L}{\partial c_t}\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right)}$

同理可得：

$dL=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^Tdh_t}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^Tdo_t\odot t a n h\left(c_t\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\right)^T{\sigma_{o_t}}^\prime\odot d U_oh_{t-1}\odot t a n h\left(c_t\right)}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\right)^T{\sigma_{o_t}}^\prime\odot d U_oh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime\right)^TdU_oh_{t-1}}\right)=tr\left(\sum_{t=1}^{T}{\left(\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime{h_{t-1}}^T\right)^TdU_o}\right)$

$\frac{\partial L}{\partial U_o}=\sum_{t=1}^{T}{\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime{h_{t-1}}^T}$

$\frac{\partial L}{\partial W_o}=\sum_{t=1}^{T}{\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime{x_t}^T}$

$\frac{\partial L}{\partial b_o}=\sum_{t=1}^{T}{\frac{\partial L}{\partial h_t}\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime}$

在简单循环神经网络 $S-RNN$ 中，我们曾给出关于矩阵 $V$ 和偏置 $b_z$ 的偏导数：

$\frac{\partial L}{\partial V}=\sum_{t=1}^{T}{\frac{\partial L}{\partial z_t}{h_t}^T}=\sum_{t=1}^{T}{\left(\widehat{y_t}-y_t\right){h_t}^T}$

$\frac{\partial L}{\partial b_z}=\sum_{t=1}^{T}\frac{\partial L}{\partial z_t}=\sum_{t=1}^{T}\left(\widehat{y_t}-y_t\right)$

记 $\delta_h^t=\frac{\partial L}{\partial h_t}$ , $\delta_c^t=\frac{\partial L}{\partial c_t}$，则有：

$\delta_c^t=\delta_h^t\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)+\delta_c^{t+1}\odot f_t$

$\delta_c^T=\delta_h^T\odot o_T\odot\left(1-{tanh}^2\left(c_T\right)\right)$

$\delta_h^{t-1}=V^T\left(\widehat{y_{t-1}}-y_{t-1}\right)+{U_i}^T\delta_h^t\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime+{U_c}^T\delta_h^t\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot i_t\odot\left(1-\widetilde{c_t}\odot\widetilde{c_t}\right)+{U_f}^T\delta_h^t\odot o_t\odot\left(1-{tanh}^2\left(c_t\right)\right)\odot c_{t-1}\odot{\sigma_{f_t}}^\prime+{U_o}^T\delta_h^t\odot tanh\left(c_t\right)\odot{\sigma_{o_t}}^\prime$

反向传播公式汇总如下：

$\frac{\partial L}{\partial U_i}=\sum_{t=1}^{T}{\delta_c^t\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime{h_{t-1}}^T}$

$\frac{\partial L}{\partial W_i}=\sum_{t=1}^{T}{\delta_c^t\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime{x_t}^T}$

$\frac{\partial L}{\partial b_i}=\sum_{t=1}^{T}{\delta_c^t\odot\widetilde{c_t}\odot{\sigma_{i_t}}^\prime}$

$\frac{\partial L}{\partial U_f}=\sum_{t=1}^{T}{\delta_c^t\odot c_{t-1}\odot{\sigma_{f_t}}^\prime{h_{t-1}}^T}$

$\frac{\partial L}{\partial W_f}=\sum_{t=1}^{T}{\delta_c^t\odot c_{t-1}\odot{\sigma_{f_t}}^\prime{x_t}^T}$

$\frac{\partial L}{\partial b_f}=\sum_{t=1}^{T}{\delta_c^t\odot c_{t-1}\odot{\sigma_{f_t}}^\prime}$

$\frac{\partial L}{\partial U_c}=\sum_{t=1}^{T}{\delta_c^t\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right){h_{t-1}}^T}$

$\frac{\partial L}{\partial W_c}=\sum_{t=1}^{T}{\delta_c^t\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right){x_t}^T}$

$\frac{\partial L}{\partial b_c}=\sum_{t=1}^{T}{\delta_c^t\odot i_t\odot\left(1-{\widetilde{c_t}}^2\right)}$

$\frac{\partial L}{\partial U_o}=\sum_{t=1}^{T}{\delta_h^t\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime{h_{t-1}}^T}$

$\frac{\partial L}{\partial W_o}=\sum_{t=1}^{T}{\delta_h^t\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime{x_t}^T}$

$\frac{\partial L}{\partial b_o}=\sum_{t=1}^{T}{\delta_h^t\odot t a n h\left(c_t\right)\odot{\sigma_{o_t}}^\prime}$

$\frac{\partial L}{\partial b_z}=\sum_{t=1}^{T}\left(\widehat{y_t}-y_t\right)$

$\frac{\partial L}{\partial V}=\sum_{t=1}^{T}{\left(\widehat{y_t}-y_t\right){h_t}^T}$

$\frac{\partial L}{\partial b_z}=\sum_{t=1}^{T}\left(\widehat{y_t}-y_t\right)$
