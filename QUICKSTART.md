# Quick Start Guide - Student Success Prediction

## 🚀 Get Running in 5 Minutes

### Prerequisites Check
```bash
# Check Python version (need 3.8+)
python --version

# Check Node.js version (need 16+)
node --version
```

---

## Step 1: Train the Model ⏱️ ~2 minutes

```bash
python group_project.py
```

**What this does:**
- Loads student data
- Trains neural network
- Saves model to `models/` folder

**Expected output:** You'll see training progress and final metrics.

---

## Step 2: Start the Backend ⏱️ ~30 seconds

### Install dependencies (first time only)
```bash
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn api:app --reload
```

**Success indicator:** You'll see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Verify:** Open http://localhost:8000/docs in your browser

---

## Step 3: Start the Frontend ⏱️ ~1 minute

### Open a new terminal window

```bash
cd frontend
```

### Install dependencies (first time only)
```bash
npm install
```

### Run the app
```bash
npm run dev
```

**Success indicator:** You'll see:
```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
```

**Verify:** Open http://localhost:5173 in your browser

---

## Step 4: Use the Application 🎉

### You should see:
- ✅ API Connected (green badge in header)
- Student form on the left
- Empty prediction panel on the right

### Try a prediction:
1. **Fill in the required fields** (marked with *)
   - First Term GPA: `3.2`
   - Second Term GPA: `3.1`
   - High School Average: `70`
   - Math Score: `63`

2. **Select some optional fields:**
   - Residency: `1 - Domestic`
   - Gender: `1 - Female`
   - Funding: `9`

3. **Click "Predict Success"**

4. **See the result** appear on the right!

5. **Click "Expand"** on the Admin Panel at the bottom to see your prediction logged

---

## 🎯 Quick Test Scenarios

### Scenario 1: Likely to Persist
```
First Term GPA: 3.5
Second Term GPA: 3.4
High School Average: 85
Math Score: 78
```

### Scenario 2: At Risk
```
First Term GPA: 2.0
Second Term GPA: 1.8
High School Average: 55
Math Score: 50
```

---

## 📊 API Endpoints Reference

Once the backend is running, try these in your browser:

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/
- **Feature Info**: http://localhost:8000/feature_columns
- **Recent Predictions**: http://localhost:8000/predictions

---

## ❌ Common Issues

### "API Connection Failed" in frontend
**Solution:** Make sure the backend is running on port 8000

### "Models directory not found"
**Solution:** Run `python group_project.py` first

### Port 8000 already in use
**Solution:** 
```bash
uvicorn api:app --reload --port 8001
```
Then update `frontend/src/api/client.js` line 3 to use port 8001

### Port 5173 already in use
**Solution:** Vite will automatically try the next available port (5174, etc.)

---

## 🛑 Stopping the Application

### Stop the Backend
Press `CTRL+C` in the backend terminal

### Stop the Frontend
Press `CTRL+C` in the frontend terminal

---

## 📝 What's Next?

1. ✅ Try different student profiles
2. ✅ Check the Admin Panel to see all predictions
3. ✅ Review the code in `frontend/src/components/`
4. ✅ Check the API documentation at `/docs`
5. ✅ Read `IMPLEMENTATION_SUMMARY.md` for technical details

---

## 🎓 For Your Presentation

### Demo Flow:
1. **Show the UI** - Clean, professional interface
2. **Enter data** - Real-time validation
3. **Get prediction** - Instant results with probabilities
4. **Show admin panel** - Structured data access API
5. **Show API docs** - http://localhost:8000/docs
6. **Explain the tech**:
   - Backend: FastAPI + PyTorch neural network
   - Frontend: React + Vite
   - Data: CSV logging
   - ML: Trained with K-fold cross-validation

### Key Points to Highlight:
- ✅ Complete full-stack application
- ✅ Modern technologies
- ✅ User-friendly interface
- ✅ Admin capabilities
- ✅ Neural network powered
- ✅ Well-documented code
- ✅ Production-ready architecture

---

## 📚 Documentation

- **Main README**: Comprehensive setup guide
- **Implementation Summary**: Technical details
- **Frontend README**: Frontend-specific docs
- **API Docs**: http://localhost:8000/docs (when running)

---

**Need Help?** Check the troubleshooting sections in README.md or the implementation summary.

**Ready to present?** You've got a complete, working full-stack ML application! 🎉

