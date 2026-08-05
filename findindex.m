T_before_test = aug_F_before(:,v_mask);
T_before_test = T_before_test(:,test_idx);

T_after_test = aug_F_after(:,v_mask);
T_after_test = T_after_test(:,test_idx);

T_before_test(:,[1 6 103])/0.00981

T_after_test(:,[1 6 103])/0.00981