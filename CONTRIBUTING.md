# Contributing to CSSP2P-GAN
We welcome contributions to improve **CSSP2P-GAN**, whether it's fixing bugs, adding features, or improving documentation. Follow the steps below to get started.

---

## 🛠️ Getting Started
1. **Fork the repository**
2. **Clone your fork locally**  
   ```bash
   git clone https://github.com/your-username/CSSP2P-GAN.git
   cd CSSP2P-GAN
   ```
3. Install dependencies
   **Option 1**: pip Environment
   ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      pip3 install --upgrade pip
      pip3 install -r requirements.txt
      ```
      
      **Option 2:** conda Environment
      ```bash
      conda env create -f environment.yaml
      conda activate stains
      ```
5. **Create a new branch for your feature or fix**
   ```bash
   git checkout -b feature/my-feature-name
   ```

## Code Guidelines
- Keep functions and modules modular and readable.
- Write docstrings and comments when needed.
- Format your code using tools like Black or prettier (optional).

## Submit a Pull Request
1. Make sure your code is clean and tested.
2. Push your branch
```bash
git push origin feature/my-feature-name
```
3. Open a **pull request** from your branch to `main`.
4. Add a clear title and description.
5. Make sure all **review comments** are resolved before requesting a merge.

## 💡Need Help?
If you find a bug or have a question, feel free to contact or open an issue.
Thank you for contributing! 🙌
