Rewrite the draft below so it reads like a person wrote it. Keep every fact and the meaning, keep it at about 180 words, and do not add new claims. Plain prose only.

Draft:
Merge pandas creates duplicates when merging orders with customers

I've an orders dataframe that has 120,000 rows approx, and customers dataframe that has 9,000 rows approx. I need to merge customer information to each order using customer_id

```python
orders.shape
 # (120000, 8) 
 customers.shape
 # (9000, 5)

merged = orders.merge(
 customers,
 on = "customer_id",
 how = "left"
)

merged.shape
 # (127845, 12)
```

So I was expecting merged to be 120,000 as each order should have only 1 customer max. But when I merge, there are extra rows, some orders are being repeated.

For example if I do:

```python
merged[merged["order_id"] == 12345]
```

I get

```text
 order_id customer_id order_date ... customer_name
 0 12345 9876 2024-01-10 ... Alice Smith
 1 12345 9876 2024-01-10 ... Alice Smith
```

Checked the unique customer_ids

```python
orders["customer_id"].nunique()
 # 9000
 customers["customer_id"].nunique()
 # 8750
```

Both Dataframes have customer_id column and data types look ok.

```python
orders["customer_id"].dtype
 # int64
 customers["customer_id"].dtype
 # int64
```

How do I figure out what is causing the extra duplicate rows when merging and make sure each order is only associated with 1 customer?

Use tools to write only the requested text to /app/answer.txt. Do not search the web. The final chat reply may briefly confirm completion.
