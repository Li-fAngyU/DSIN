mkdir big_train
mkdir big_test
wget -O model_input.tar.gz https://bj.bcebos.com/v1/ai-studio-online/1efae462a1ac4f5483c680fb5f9396038a85eb5227314c878789acefc3850491?responseContentDisposition=attachment%3B%20filename%3Dmodel_input.tar.gz&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-03-13T10%3A24%3A59Z%2F-1%2F%2F1201620aa0733d0cf1f300df5d7b3029b32ed101df823eb0e8211c667b61732b
tar -zxvf model_input.tar.gz -C big_train
mv big_train/test_feat_input.pkl big_test/  
mv big_train/test_label.pkl big_test/  
mv big_train/test_sess_input.pkl big_test/  
mv big_train/test_session_length.pkl big_test/