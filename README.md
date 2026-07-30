# Bomberland AI Agents

Dự án mô phỏng Bomberland gồm ba AI agent với ba chiến thuật khác nhau, được xây
dựng cho GDGoC AI Challenge.

- **Agent 1 — PPO Agent:** học bằng reinforcement learning, kết hợp safety shield.
- **Agent 2 — Defensive Agent:** phòng thủ, tính vùng nguy hiểm và tìm đường bằng BFS.
- **Agent 3 — Aggressive Agent:** chủ động săn đối thủ bằng A* và đặt bom gây áp lực.

## Video demo

![Ba Bomberland AI agent thi đấu](assets/demo-longest.gif)

## Kết quả đánh giá PPO

Agent 1 được đánh giá qua 30 trận đấu với Defensive Agent và Aggressive Agent.
Mỗi trận có tối đa 200 Step.

| Chỉ số | Kết quả |
|---|---:|
| Tỷ lệ thắng | **36,7%** — 11/30 trận |
| Tỷ lệ sống sót | **40%** — 12/30 trận |
| Điểm trung bình | **24,7** |
| Tổng hành động đặt bom | **178** |

Reward của PPO khuyến khích agent:

- Né bom và thoát khỏi vùng nguy hiểm.
- Phá thùng để mở đường.
- Tiếp cận và săn đối thủ.
- Chọn thời điểm đặt bom có mục tiêu.
- Sống sót và trở thành agent cuối cùng.

### Bảng reward

| Sự kiện | Reward |
|---|---:|
| Sống qua mỗi Step | **+0,02** |
| Thoát khỏi vùng nguy hiểm | **+0,20** |
| Di chuyển gần đối thủ hơn | **+0,03** |
| Đặt bom có thể trúng thùng hoặc đối thủ | **+0,30** |
| Phá một thùng | **+1,00** |
| Hạ một đối thủ | **+10,00** |
| Trở thành agent cuối cùng | **+15,00** |
| Đi vào vùng nguy hiểm | **−0,35** |
| Hành động nhưng vị trí không thay đổi | **−0,02** |
| Bị loại | **−10,00** |

Điểm phá thùng và hạ đối thủ được chuyển thành reward theo công thức:

```python
reward += (new_score - old_score) * 0.10
```

Engine cộng 10 điểm khi phá thùng và 100 điểm khi hạ đối thủ, tương ứng với
`+1,00` và `+10,00` reward.

## Chạy đấu trường trên web

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Khởi động web:

```bash
python app.py
```

Sau đó mở:

```text
http://127.0.0.1:8001
```

## Huấn luyện PPO

Huấn luyện model mới:

```bash
python train_ppo.py --matches 1000 --output bomberland_ppo.pt
```

Tiếp tục huấn luyện từ checkpoint hiện có:

```bash
python train_ppo.py --matches 500 --output bomberland_ppo.pt --resume
```

## Chạy engine trong terminal

```bash
python local_engine.py --steps 200 --show-map
```

## Tạo lại GIF demo

```bash
python record_demo.py
```
