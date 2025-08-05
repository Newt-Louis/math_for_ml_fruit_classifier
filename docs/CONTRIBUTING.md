### Configure Kaggle API Authentication

In order for this project to be able to download data from Kaggle, you need to configure the API authentication file on your computer.

1. **Download the `kaggle.json` file**
   * Log in to your Kaggle account.
   * Go to your **Account** page at: `https://www.kaggle.com/account`
   * Click the **Settings** button and scroll down to the **Create New API Token** section.
   * The browser will automatically download the `kaggle.json` file.

2. **Put the file in the right location**
You need to move the downloaded `kaggle.json` file to the right folder so that the Kaggle library can find it.

* **On macOS / Linux:**
```bash
# Create a .kaggle folder if it doesn't exist and move the file into it
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/

# Grant secure access (only you can read and write)
chmod 600 ~/.kaggle/kaggle.json
```

* **On Windows:**
You need to move the `kaggle.json` file into the `C:\Users\<User-Name>\.kaggle\` folder.

  1. Open File Explorer.
  2. Go to `C:\Users\<User-Name>\`.
  3. Create a new folder called `.kaggle`.
  4. Copy the `kaggle.json` file you just downloaded and paste it into this `.kaggle` folder.

## Documentation

For more detailed information, please refer to the following guides:

* [Data Handling Guide](./DATA_GUIDE.md)
* [Model Training Guide](./TRAINING_GUIDE.md)