#!/usr/bin/env python3
"""预处理 GSM8K 数据集，将 answer 添加到 messages 中"""

import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_parquet('/root/datasets/gsm8k/train.parquet')

# 为每行添加 assistant 回答
def add_assistant_message(row):
    # messages 是 numpy.ndarray，转换为列表
    messages = row['messages'].tolist() if isinstance(row['messages'], np.ndarray) else list(row['messages'])
    # 添加 assistant 的回答
    messages.append({
        'role': 'assistant',
        'content': row['answer']  # 使用完整的推理过程
    })
    return messages

df['messages'] = df.apply(add_assistant_message, axis=1)

# 保存处理后的数据
output_path = '/root/datasets/gsm8k/train_with_answers.parquet'
df.to_parquet(output_path)

print(f"✅ 处理完成！保存到: {output_path}")
print(f"总共处理了 {len(df)} 条数据")
print("\n示例数据（前 3 个 message）:")
for i, msg in enumerate(df.iloc[0]['messages']):
    print(f"\nMessage {i+1} ({msg['role']}):")
    content = msg['content']
    if len(content) > 100:
        print(f"  {content[:100]}...")
    else:
        print(f"  {content}")

