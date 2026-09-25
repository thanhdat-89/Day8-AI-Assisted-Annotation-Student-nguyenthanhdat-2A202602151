# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

**Họ và tên:** Nguyễn Thành Đạt

**Công cụ gán nhãn đã dùng:** CVAT

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian và có vùng đệm ở giữa nhằm giảm nguy cơ rò rỉ dữ liệu giữa hai tập. Với dữ liệu video, các frame gần nhau theo thời gian thường rất giống nhau về vị trí xe, ánh sáng, góc camera và bố cục cảnh. Nếu chia ngẫu nhiên từng frame, một ảnh trong test có thể gần như trùng với một ảnh trong pool/train. Khi đó mô hình được đánh giá trên dữ liệu quá giống dữ liệu đã dùng trong quá trình học, khiến kết quả test có xu hướng lạc quan hơn khả năng tổng quát hóa thực tế.

Việc chia theo thời gian và chừa vùng đệm làm cho test độc lập hơn với dữ liệu dùng cho Active Learning. Trong bài này, tập test gồm **20 ảnh với 403 box tham chiếu**, trong đó **14 box có chiều cao dưới 16 px được bỏ qua khi đánh giá**. Các chỉ số P, R và F1 được tính tại confidence 0.25, còn việc ghép prediction với reference sử dụng IoU 0.5.

## 2. Mô hình khởi đầu lạnh (cold start)

Kết quả vòng 0:

| vòng | model | ảnh train | box train | AP50 | Delta AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | - | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Các số liệu cho thấy cold start có **precision cao (0.925) nhưng recall chỉ 0.489**. Nói cách khác, những xe mô hình quyết định phát hiện thường khá chính xác, nhưng mô hình vẫn bỏ sót nhiều xe.

Điểm yếu thể hiện đặc biệt rõ theo kích thước. Recall của xe nhỏ chỉ **0.182**, thấp hơn nhiều so với xe trung bình **0.547** và xe lớn **0.561**. Điều này phù hợp với đặc điểm của cảnh đường cao tốc ban đêm: xe ở xa thường nhỏ, tối, có thể chỉ nổi bật bằng cụm đèn và dễ bị mô hình bỏ sót.

Quan sát `compare_round0.jpg` cũng cho thấy các trường hợp khó tập trung ở xe nhỏ/xa, xe tối và các xe nằm trong cụm phương tiện. Tuy nhiên, không nên tự động xem mọi khác biệt với reference là lỗi của mô hình. Chẳng hạn, đối với xe rất xa chỉ còn cụm đèn hoặc vật thể bị cắt ở mép ảnh, cần người rà lại reference theo `GUIDELINE_LABEL.md` trước khi kết luận prediction sai. Bản thân nhãn test cũng do mô hình tạo và chưa được người kiểm tra lại từng box.

## 3. Chiến lược chọn mẫu

Điểm chọn mẫu được tính theo:

```text
score = W_U * U + W_A * A + W_D * D
```

Trong đó, **U (uncertainty)** phản ánh mức độ mô hình không chắc chắn về prediction; **A** phản ánh mức độ xuất hiện các prediction mơ hồ cần người kiểm tra; và **D (diversity)** hỗ trợ ưu tiên mẫu có giá trị bổ sung thay vì chỉ lấy nhiều frame gần giống nhau. Trọng số `W_U`, `W_A`, `W_D` quyết định mức đóng góp của từng thành phần vào điểm cuối.

`MIN_GAP_S` đóng vai trò tạo khoảng cách thời gian tối thiểu giữa các frame được chọn. Điều này đặc biệt quan trọng với video vì hai frame cách nhau rất ngắn có thể gần như giống hệt nhau. Nếu chỉ xếp hạng theo uncertainty mà không xét khoảng cách thời gian, ngân sách annotation có thể bị tiêu tốn cho nhiều ảnh gần trùng.

Từ `selection_round1.csv`, có thể thấy một số ví dụ:

- `frame_0182.jpg` đứng hạng 1 với **score 0.9591**, U = **0.9182**, A = **1.0000**, 28 box và 18 prediction mơ hồ; frame này được chọn.
- `frame_0369.jpg` đứng hạng 2 với **score 0.9324**, U = **0.9315**, 43 box và 16 prediction mơ hồ; frame này được chọn.
- `frame_0380.jpg` đứng hạng 3 với **score 0.9170**, U = **0.9340**, 40 box và 15 prediction mơ hồ; frame này được chọn.
- `frame_0372.jpg` vẫn có score rất cao **0.9101** và U = **0.9202**, nhưng không được chọn. Frame này ở thời điểm 148.8 s, rất gần `frame_0369.jpg` ở 147.6 s và `frame_0380.jpg` ở 152.0 s. Đây là ví dụ cho thấy việc chọn mẫu không chỉ lấy các frame có uncertainty cao nhất mà còn phải hạn chế ảnh gần trùng để sử dụng công gán nhãn hiệu quả.

Điểm uncertainty cao **không chứng minh** rằng annotation ảnh đó chắc chắn sẽ cải thiện mô hình. Nó chỉ cho biết model hiện tại đang không chắc chắn. Ảnh có thể khó do ánh sáng, nhiễu, nhiều vật thể nhỏ hoặc gần trùng với dữ liệu đã có. Giá trị thực sự của mẫu chỉ có thể đánh giá sau khi người rà nhãn, fine-tune và đo lại trên tập test độc lập.

## 4. Các vòng học chủ động (Active Learning)

Kết quả hai vòng:

| vòng | model | ảnh train | box train | AP50 | Delta AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | - | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 338 | 0.496 | -0.276 | 1.000 | 0.109 | 0.197 | 0.000 | 0.064 | 0.610 |

Ở vòng 1, tôi rà **12 ảnh** bằng CVAT. Kết quả so sánh pre-label và annotation sau khi sửa được công cụ báo trực tiếp là **143 box giữ nguyên (accepted), 14 box chỉnh sửa (edited), 12 box xóa (deleted) và 181 box thêm mới (added)**, tạo thành bộ train cuối gồm **338 box**. Số lượng box thêm mới lớn cho thấy pre-label ban đầu bỏ sót đáng kể các đối tượng trong batch được Active Learning lựa chọn.

`REVIEW_LOG.csv` ghi lại ba trường hợp cụ thể ở `frame_0099.jpg`: một xe lớn sát mép dưới khung ảnh được **added** vì vẫn nhìn thấy thân xe và xác định được ranh giới theo guideline; một box chứa hai xe được **edited** vì hai xe không được gộp chung một bounding box; và một prediction sát mép phải được **deleted** vì không đủ thông tin để xác định vật thể là `car`.

Blind Scan được thực hiện trên `frame_0107.jpg`, trong đó ghi nhận **26 xe bằng quan sát mắt**. Một trường hợp được xác định là dễ bị AI bỏ sót là xe sát mép dưới ảnh do vật thể bị cắt và không có đèn xe rõ ràng. Trường hợp này phù hợp với guideline rằng xe bị cắt ở mép ảnh vẫn được gán nhãn nếu có thể xác định là xe, nhưng box chỉ bao phần nằm trong ảnh. Lưu ý: trong `BLIND_SCAN.md`, hai mô tả vị trí dễ sai hiện đang trùng nhau; vì file đã được khóa trong `blind_lock.json`, tôi giữ nguyên nội dung này khi viết báo cáo và chỉ dùng nó như bằng chứng về một dạng ca khó.

Sau fine-tune, kết quả tổng thể **không cải thiện**. AP50 giảm từ **0.771 xuống 0.496**, tức giảm **0.276** so với cold start. Precision tại confidence 0.25 tăng từ **0.925 lên 1.000**, nhưng recall giảm rất mạnh từ **0.489 xuống 0.109**, kéo F1 từ **0.640 xuống 0.197**. Điều này cho thấy Round 1 trở nên rất bảo thủ: các prediction còn lại có độ chính xác cao, nhưng mô hình bỏ sót quá nhiều xe.

Sự thay đổi theo kích thước còn rõ hơn. Recall small giảm **0.182 -> 0.000**, recall medium giảm **0.547 -> 0.064**, trong khi recall large lại tăng **0.561 -> 0.610**. Vì vậy, fine-tune không làm mọi nhóm xe xấu đi như nhau: **xe lớn tốt lên nhẹ, còn xe nhỏ và trung bình giảm mạnh**.

`compare_round1.jpg` cung cấp bằng chứng trực quan cho hiện tượng này. Ví dụ ở `frame_0050`, cold start đạt **TP=11, FP=2, FN=7**, còn Round 1 chỉ còn **TP=4, FP=0, FN=14**. Ở `frame_0150`, cold start là **TP=10, FP=2, FN=10**, trong khi Round 1 là **TP=1, FP=0, FN=19**. Như vậy Round 1 loại bỏ false positive nhưng đồng thời bỏ sót nhiều xe hơn đáng kể, phù hợp với sự tăng precision và sụt recall trong metrics tổng thể.

Ba nguồn thông tin cần được phân biệt: **Blind Scan** ghi nhận quan sát độc lập của người rà trước khi xem pre-label; **REVIEW_LOG và kết quả sửa CVAT** mô tả lỗi của pre-label và quyết định annotation; còn **compare_round1.jpg/metrics_round1** phản ánh hành vi của model sau fine-tune. Việc sửa được nhiều lỗi pre-label không đồng nghĩa model sau một vòng fine-tune nhỏ chắc chắn sẽ tốt hơn.

## 5. Kết luận và giới hạn

Sau một vòng Active Learning, kết quả tổng thể **thấp hơn cold start** trên tập test hiện tại. AP50 giảm từ **0.771 xuống 0.496 (-0.276)** và recall giảm từ **0.489 xuống 0.109**, mặc dù precision tăng lên **1.000**. Điểm đáng chú ý là recall xe lớn tăng từ **0.561 lên 0.610**, nhưng không bù được mức giảm rất mạnh của xe nhỏ và trung bình.

Với kết quả này, tôi sẽ **chưa tiếp tục fine-tune ngay** chỉ với mục tiêu thêm dữ liệu. Trước tiên cần kiểm tra nguyên nhân khiến model trở nên quá bảo thủ sau Round 1, đặc biệt vì batch huấn luyện chỉ gồm **12 ảnh và 338 box**, trong khi phân bố kích thước đối tượng có thể chưa đại diện đầy đủ cho tập test.

Nếu thực hiện vòng 2, tôi ưu tiên các ca **xe nhỏ/xa trong cảnh tối hoặc cụm nhiều xe**, vì recall small hiện bằng 0; đồng thời ưu tiên các **xe trung bình có ánh sáng yếu, bị che hoặc khó phân biệt với ánh đèn**, vì recall medium chỉ còn 0.064. Khi chọn các ảnh này cần cân nhắc chi phí annotation: cảnh có nhiều xe nhỏ cần kiểm tra từng box kỹ hơn và ranh giới vật thể khó xác định. Ngoài ra phải tiếp tục áp dụng khoảng cách thời gian để tránh tốn ngân sách vào các frame gần trùng nhau.

Kết luận của bài có ba giới hạn quan trọng. Thứ nhất, test chỉ có **20 ảnh**, nên thay đổi trên một số ít frame có thể ảnh hưởng đáng kể đến metric. Thứ hai, **14 box cao dưới 16 px được bỏ qua**, do đó metric không phản ánh toàn bộ khả năng phát hiện các xe cực nhỏ. Thứ ba, **403 box tham chiếu chưa được người rà thủ công toàn bộ**, nên AP50 ở đây đo mức khớp với reference hiện tại chứ chưa thể xem là thước đo tuyệt đối về khả năng nhận diện xe ngoài thực tế.

Nếu AP50 giảm như Round 1, trước khi train thêm tôi sẽ kiểm tra lại **chất lượng và tính nhất quán của annotation, phân bố kích thước xe trong 12 ảnh train, các trường hợp added/edited/deleted, mức độ gần trùng giữa các frame được chọn và các hyperparameter fine-tune**. Sau đó mới quyết định cần sửa dữ liệu, bổ sung batch Active Learning cân bằng hơn hay điều chỉnh quá trình fine-tune. Việc train thêm ngay khi chưa xác định nguyên nhân có nguy cơ làm mô hình tiếp tục tối ưu theo một batch nhỏ và làm recall giảm thêm.
