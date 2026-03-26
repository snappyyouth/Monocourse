"""
仅使用 PyTorch 基础张量运算实现情感分类。
禁止使用 torch.nn / torch.optim / torch.utils.data，所有逻辑手动实现。
数据格式: text\tlabel (标签 0-4)
"""

import csv
import os
import torch
from collections import Counter


# ===================== 1. 读取 TSV =====================
def read_tsv(file_path):
    """读取 TSV 文件，返回 (词列表的列表, 标签列表)"""
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                texts.append(row[0].strip().lower().split())
                labels.append(int(row[1]))
    return texts, labels


# ===================== 2. 构建词表 =====================
def build_vocab(texts, max_size=5000):
    """统计词频，保留高频词，返回 word -> index 映射"""
    # TODO 酒店中NextFind中的词表跟这个有关系吗？
    word_counts = Counter(w for tokens in texts for w in tokens)
    vocab = {"<unk>": 0}
    for word, _ in word_counts.most_common(max_size):
        vocab[word] = len(vocab)
    return vocab


# ===================== 3. 文本 -> 词袋向量 =====================
def texts_to_bow(texts, vocab):
    """将文本列表转换为词袋矩阵 (num_samples, vocab_size)"""
    n = len(texts)
    v = len(vocab)
    bow = torch.zeros(n, v)
    for i, tokens in enumerate(texts):
        for w in tokens:
            idx = vocab.get(w, vocab["<unk>"])
            bow[i, idx] += 1.0
    return bow


# ===================== 4. Softmax（手动实现）=====================
def softmax(logits):
    """数值稳定的 softmax: (batch, num_classes) -> (batch, num_classes)"""
    max_val = logits.max(dim=1, keepdim=True).values
    exp = torch.exp(logits - max_val)
    return exp / exp.sum(dim=1, keepdim=True)


# ===================== 5. 交叉熵损失（手动实现）=====================
def cross_entropy_loss(probs, labels):
    """
    probs:  (batch, num_classes) softmax 输出
    labels: (batch,) 整数标签
    返回: 标量平均损失
    """
    n = probs.shape[0]
    # 取出每个样本对应正确类别的概率
    correct_probs = probs[torch.arange(n), labels]
    return -torch.log(correct_probs + 1e-12).mean()


# ===================== 6. 前向传播 =====================
def forward(X, W, b):
    """
    线性分类器: logits = X @ W + b
    X: (batch, vocab_size)
    W: (vocab_size, num_classes)
    b: (num_classes,)
    返回: (logits, probs)
    """
    logits = X @ W + b          # 矩阵乘法 + 广播加偏置
    probs = softmax(logits)
    return logits, probs


# ===================== 7. 反向传播（手动求梯度）=====================
def backward(X, probs, labels, W):
    """
    softmax + cross_entropy 的梯度:
      dL/d(logits) = probs - one_hot(labels)
      dL/dW = X^T @ dL/d(logits)
      dL/db = sum(dL/d(logits), dim=0)

    返回: (grad_W, grad_b)
    """
    n = X.shape[0]
    num_classes = probs.shape[1]

    # dL/d(logits) = probs - one_hot(labels)
    one_hot = torch.zeros(n, num_classes)
    one_hot[torch.arange(n), labels] = 1.0
    d_logits = (probs - one_hot) / n    # 除以 n 对应 mean loss

    grad_W = X.t() @ d_logits            # (vocab_size, num_classes)
    grad_b = d_logits.sum(dim=0)          # (num_classes,)
    return grad_W, grad_b


# ===================== 8. 评估准确率 =====================
def accuracy(probs, labels):
    preds = probs.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ===================== 9. 保存预测结果到 TSV =====================
def save_predictions(texts, labels, preds, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "true_label", "pred_label"])
        for tokens, true_l, pred_l in zip(texts, labels, preds):
            writer.writerow([" ".join(tokens), true_l, pred_l])
    print(f"预测结果已保存到 {output_path}")


# ===================== 10. 主流程 =====================
def main():
    # ---------- 读取数据 ----------
    _dir = os.path.dirname(os.path.abspath(__file__))
    train_texts, train_labels = read_tsv(os.path.join(_dir, "new_train.tsv"))
    test_texts, test_labels = read_tsv(os.path.join(_dir, "new_test.tsv"))
    print(f"训练集: {len(train_labels)} 条, 测试集: {len(test_labels)} 条")

    # ---------- 构建词表 & 词袋向量 ----------
    vocab = build_vocab(train_texts, max_size=5000)
    print(f"词表大小: {len(vocab)}")

    X_train = texts_to_bow(train_texts, vocab)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = texts_to_bow(test_texts, vocab)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    # ---------- 初始化参数（Xavier 初始化）----------
    num_classes = 5
    vocab_size = len(vocab)
    std = (2.0 / (vocab_size + num_classes)) ** 0.5
    W = torch.randn(vocab_size, num_classes) * std
    b = torch.zeros(num_classes)

    # ---------- 训练（mini-batch SGD）----------
    lr = 0.01
    batch_size = 64
    num_epochs = 30
    n_train = X_train.shape[0]

    for epoch in range(1, num_epochs + 1):
        # 随机打乱训练集
        perm = torch.randperm(n_train)
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # 前向
            logits, probs = forward(X_batch, W, b)
            loss = cross_entropy_loss(probs, y_batch)

            # 反向
            grad_W, grad_b = backward(X_batch, probs, y_batch, W)

            # 参数更新（SGD）
            W = W - lr * grad_W
            b = b - lr * grad_b

            epoch_loss += loss.item() * (end - start)
            epoch_correct += (probs.argmax(dim=1) == y_batch).sum().item()

        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train

        # 测试集评估
        _, test_probs = forward(X_test, W, b)
        test_loss = cross_entropy_loss(test_probs, y_test).item()
        test_acc = accuracy(test_probs, y_test)

        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
        )

    # ---------- 保存预测结果 ----------
    _, test_probs = forward(X_test, W, b)
    test_preds = test_probs.argmax(dim=1).tolist()
    save_predictions(
        test_texts, test_labels, test_preds, "pytorch_practice/predictions.tsv"
    )

    # ---------- 保存模型参数 ----------
    torch.save({"W": W, "b": b, "vocab": vocab}, "pytorch_practice/model.pt")
    print("模型参数已保存到 pytorch_practice/model.pt")


if __name__ == "__main__":
    main()
