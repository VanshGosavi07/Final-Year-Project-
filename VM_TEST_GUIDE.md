# VM Testing Guide - Final Year Project

## ✅ Pre-Deployment Verification Checklist

Run these commands on your VM (flask-ml-vm) to verify everything works before deploying to Cloud Run.

---

## 🔧 Step 1: Pull Latest Code

```bash
cd ~/flask_app/Final-Year-Project-
git pull origin main
```

**Expected output:**
- "Already up to date." OR
- "Updating 825f248..55cf50e" (showing the new commit)
- Files changed: breast_cancer_predictor.py, lung_cancer_predictor.py, main.py, etc.

**Latest commit should be:** `55cf50e - Return explicit is_malignant flag and robust visualization checks`

---

## 🧪 Step 2: Quick Verification Tests

### Test 2A: Model Prediction + Visualization
This verifies that:
- Models load correctly
- Predictions work
- Annotated images are created with correct colored borders (green for non-malignant, red for malignant)

```bash
cd ~/flask_app/Final-Year-Project-
python verify_visualization.py
```

**Expected output:**
```
Copied sample image to static/uploads/test_visualization_input.jpg
INFO:main:Loading Breast Cancer Predictor (first time - may take 10-30 seconds)...
INFO:breast_cancer_predictor:Loading Breast Cancer model from: Modal/Breast Cancer/breast_cancer.keras (31.5 MB)
INFO:breast_cancer_predictor:Breast Cancer model loaded successfully!
INFO:main:Breast Cancer Predictor loaded ✓
model_predict results: [('Cancer: No (Non-Malignant)', 'uploads/test_visualization_input_output.png')]
Output image: static/uploads/test_visualization_input_output.png
Output image size: 553x619
Expected rectangle: (100, 100, 453, 519)
Visualization check PASSED: found color (0, 255, 0) on border for label Cancer: No (Non-Malignant)
```

**✅ Success criteria:** "Visualization check PASSED"

**❌ If fails:** Check that Git LFS pulled the real model files (should be ~31.5 MB for breast, ~148 MB for lung)

---

### Test 2B: Report Template Rendering
Verifies the report HTML includes the annotated image path correctly.

```bash
cd ~/flask_app/Final-Year-Project-
python render_report_check.py
```

**Expected output:**
```
Rendered HTML contains expected image path: True
Snippet: 
          <!-- Local file path -->
          <img
            src="/static/uploads/test_visualization_input_output.png"
            alt="Medical Scan Image"
```

**✅ Success criteria:** "Rendered HTML contains expected image path: True"

---

## 🚀 Step 3: Start Flask App (Performance Mode)

Start the app to verify full end-to-end flow:

```bash
cd ~/flask_app/Final-Year-Project-
python main.py
```

**Expected startup logs:**
```
================================================================================
🚀 PERFORMANCE MODE: Pre-loading all models and RAG data...
================================================================================

📚 Step 1: Loading RAG Document Processors...
--------------------------------------------------------------------------------
✓ General processor loaded
✓ Breast cancer RAG processor loaded with PDF data
✓ Lung cancer RAG processor loaded with PDF data

🤖 Step 2: Loading ML Prediction Models...
--------------------------------------------------------------------------------
✓ Breast cancer ML model loaded (31.5 MB)
✓ Lung cancer ML model loaded (148.1 MB)

================================================================================
🎉 ALL MODELS PRE-LOADED! Startup time: 67.XX seconds
⚡ Application ready for INSTANT responses!
================================================================================
```

**⏱️ Startup time:** ~60-80 seconds (one-time; after this, all requests are instant)

---

## 🌐 Step 4: Manual Browser Test

1. **Access the app** (use VM external IP or localhost):
   ```
   http://localhost:5000
   ```
   OR
   ```
   http://34.170.30.162:5000  (replace with your VM IP)
   ```

2. **Login/Register** and navigate to the diagnosis form

3. **Generate a test report:**
   - Fill patient details
   - Select disease: "Breast Cancer" or "Lung Cancer"
   - Upload a medical scan image
   - Submit

4. **Verify in the report page:**
   - ✅ Annotated image is displayed (should have colored rectangle around it)
   - ✅ Rectangle color: **Red** for malignant, **Green** for non-malignant/normal
   - ✅ Label text appears on the image
   - ✅ Report includes patient details, recommendations, etc.

5. **Test chat feature:**
   - Click "Chat With Report"
   - Ask questions like:
     - "What is the patient's name?"
     - "What is the diagnosis?"
     - "What diet should I follow?"
   - ✅ Chat should respond instantly (no delay after startup)
   - ✅ Responses should include patient-specific information

---

## 📊 Step 5: Check Logs

While the app is running, check for any errors:

```bash
# In another terminal
tail -f ~/flask_app/Final-Year-Project-/logs/*.log
```

**Look for:**
- ❌ Any ERROR or CRITICAL messages
- ✅ Successful prediction logs
- ✅ No model loading errors

---

## 🔍 Step 6: Verify Output Images

Check that annotated images are being saved:

```bash
cd ~/flask_app/Final-Year-Project-
ls -lh static/uploads/*_output.png
```

**Expected:**
- One or more `*_output.png` files
- File sizes should be reasonable (few hundred KB)

**Open an output image to inspect visually:**
```bash
# Copy to local machine and open, or use:
file static/uploads/test_visualization_input_output.png
```

Should show: PNG image data

---

## 🧹 Cleanup Test Files (Optional)

After testing, remove test files:

```bash
cd ~/flask_app/Final-Year-Project-
rm static/uploads/test_visualization_input*
```

---

## ✅ All Tests Passed? Ready to Deploy!

If all the above steps pass:

### Deploy to Cloud Run

```bash
cd ~/flask_app/Final-Year-Project-
bash deploy_with_secrets.sh
```

**Deployment checklist:**
- ✅ Git LFS models are in the repo
- ✅ All environment variables in `.env` file
- ✅ Cloud Run configured with:
  - Memory: 2Gi minimum
  - Min instances: 1 (for instant responses)
  - Timeout: 300s
  - CPU: 2

**Expected deployment time:** 3-5 minutes

**After deployment:**
1. Test the Cloud Run URL (provided at end of deploy script)
2. Run the same manual browser test on the live URL
3. Monitor Cloud Run logs for any errors

---

## 🐛 Troubleshooting

### Issue: "Model Unavailable"
**Solution:** Run `git lfs pull` to get real model files (not pointers)

### Issue: "Visualization check FAILED"
**Solution:** Check that predictors return correct tuple format. Verify by running:
```bash
python -c "from main import get_breast_cancer_predictor; p = get_breast_cancer_predictor(); print(p.predict_with_visualization('Modal/Breast Cancer/Dataset/test/0/0.jpg', 'test_out.png'))"
```

### Issue: Chat is slow
**Solution:** Ensure PERFORMANCE MODE is enabled in `main.py` (should be default now)

### Issue: Images not displaying in report
**Solution:** 
1. Check `static/uploads/` folder exists
2. Verify paths in report HTML: `view-source:http://localhost:5000/report`
3. Ensure `url_for('static', filename=...)` is used in template

---

## 📝 Notes

- **First request after startup:** Should be instant (models pre-loaded)
- **Subsequent requests:** Near-instant (all cached)
- **Memory usage:** ~1.5-2 GB during operation
- **Startup time:** One-time cost of ~60-80 seconds

---

## 🎯 Success Criteria Summary

| Test | Expected Result |
|------|----------------|
| Git pull | Latest commit (55cf50e) |
| Model verification | ✅ PASSED |
| Report rendering | ✅ Image path found in HTML |
| Flask startup | ✅ Models pre-loaded in ~60-80s |
| Browser test | ✅ Annotated image shown with colored border |
| Chat test | ✅ Instant responses with patient data |
| Deployment | ✅ Cloud Run URL accessible |

---

**Last Updated:** November 10, 2025  
**Commit:** 55cf50e
