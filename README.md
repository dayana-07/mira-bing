# `README.md` for [mira-bing](https://github.com/Ai-Yukino/mira-bing)

{Fun image goes here}

---

## ❄ Setup ❄

### ⚪ Clone this repo

Open your prefered terminal. Navigate to where you want to clone this repo, e.g.

```
cd documents
```

Run

```
git clone https://github.com/Ai-Yukino/mira-bing
```

Navigate inside the repo with

```
cd mira-bing
```

### ⚫ Install virtual environment

Run

```
conda env create -f mira-bing.yml
```

### ⚪ Activate virtual envrionment

Run

```
conda activate mira-bing
```

## 🌸 Generate Jupyter notebook 🌸

### ⚪ Navigate to notebook folder

Run

```
cd notebooks/01
```

### ⚫ Generate notebook

Run

```
jupytext --to ipynb 01.py
```

to generate the notebook as a `01.ipynb` file.

## ❄ Open Jupyter notebook ❄

Run

```
jupyter lab --no-browser
```

and then click one of the links that pops up in your terminal to open the notebook in Jupyter Lab.

<center>
<img src="images/lab.png" style="border-radius: 10px">
</center>
