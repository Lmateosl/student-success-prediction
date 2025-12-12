import { useState, useEffect } from 'react';
import { getFeatureColumns } from '../api/client';
import HelpIcon from './HelpIcon';
import './StudentForm.css';

function StudentForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState({
    'First Term GPA': '',
    'Second Term GPA': '',
    'High School Average': '',
    'Math Score': '',
    'First Language': '',
    'Funding': '',
    'School': '',
    'Fast Track': '',
    'Coop': '',
    'Residency': '',
    'Gender': '',
    'Prev Education': '',
    'Age Group': '',
    'English Grade': '',
  });

  const [categoricalOptions, setCategoricalOptions] = useState({});
  const [errors, setErrors] = useState({});

  useEffect(() => {
    // Fetch categorical field options from the API
    async function fetchMetadata() {
      try {
        const metadata = await getFeatureColumns();
        const options = {};
        
        metadata.raw_categorical_fields.forEach((field) => {
          options[field.raw_name] = field.valid_codes;
        });
        
        setCategoricalOptions(options);
      } catch (error) {
        console.error('Failed to load field metadata:', error);
      }
    }
    
    fetchMetadata();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    
    // Clear error for this field when user starts typing
    if (errors[name]) {
      setErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[name];
        return newErrors;
      });
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    // Validate numeric fields (required)
    const numericFields = ['First Term GPA', 'Second Term GPA', 'High School Average', 'Math Score'];
    numericFields.forEach((field) => {
      if (!formData[field] || formData[field] === '') {
        newErrors[field] = 'This field is required';
      } else if (isNaN(formData[field])) {
        newErrors[field] = 'Must be a number';
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    // Convert form data to proper types
    const processedData = {};
    
    // Process numeric fields
    ['First Term GPA', 'Second Term GPA', 'High School Average', 'Math Score'].forEach((field) => {
      if (formData[field] !== '') {
        processedData[field] = parseFloat(formData[field]);
      }
    });
    
    // Process categorical fields
    ['First Language', 'Funding', 'School', 'Fast Track', 'Coop', 'Residency', 
     'Gender', 'Prev Education', 'Age Group', 'English Grade'].forEach((field) => {
      if (formData[field] !== '') {
        processedData[field] = parseInt(formData[field]);
      }
    });
    
    onSubmit(processedData);
  };

  const getCategoricalLabel = (fieldName, code) => {
    // Official definitions from the assignment
    const labels = {
      'Residency': { 1: 'Domestic', 2: 'International' },
      'Gender': { 1: 'Female', 2: 'Male', 3: 'Neutral' },
      'Fast Track': { 1: 'Yes', 2: 'No' },
      'Coop': { 1: 'Yes', 2: 'No' },
      'Prev Education': { 1: 'HighSchool', 2: 'PostSecondary' },
      'First Language': { 1: 'English', 2: 'French', 3: 'Other' },
      'School': {
        1: 'Advancement',
        2: 'Business',
        3: 'Communications',
        4: 'Community and Health',
        5: 'Hospitality',
        6: 'Engineering',
        7: 'Transportation'
      },
      'Funding': {
        1: 'Apprentice_PS',
        2: 'GPOG_FT',
        3: 'Intl Offshore',
        4: 'Intl Regular',
        5: 'Intl Transfer',
        6: 'Joint Program Ryerson',
        7: 'Joint Program UTSC',
        8: 'Second Career Program',
        9: 'Work Safety Insurance Board'
      },
      'Age Group': {
        1: '0 to 18',
        2: '19 to 20',
        3: '21 to 25',
        4: '26 to 30',
        5: '31 to 35',
        6: '36 to 40',
        7: '41 to 50',
        8: '51 to 60',
        9: '61 to 65',
        10: '66+'
      },
      'English Grade': {
        1: 'Level-130',
        2: 'Level-131',
        3: 'Level-140',
        4: 'Level-141',
        5: 'Level-150',
        6: 'Level-151',
        7: 'Level-160',
        8: 'Level-161',
        9: 'Level-170',
        10: 'Level-171',
        11: 'Level-180'
      }
    };
    
    if (labels[fieldName] && labels[fieldName][code]) {
      return `${code} - ${labels[fieldName][code]}`;
    }
    return code;
  };

  return (
    <form className="student-form" onSubmit={handleSubmit}>
      <h2>Student Information</h2>
      
      {/* Academic Performance Section */}
      <div className="form-section">
        <h3>📊 Academic Performance</h3>
        
        <div className="form-group">
          <label htmlFor="first-term-gpa">
            First Term GPA <span className="required">*</span>
            <HelpIcon text="The student's Grade Point Average (GPA) from their first term of study. Valid range: 0.0 to 4.5. This is a strong indicator of early academic performance. Higher GPAs suggest better academic success and persistence." />
          </label>
          <input
            id="first-term-gpa"
            type="number"
            step="0.01"
            name="First Term GPA"
            value={formData['First Term GPA']}
            onChange={handleChange}
            placeholder="0.0 - 4.0"
            className={errors['First Term GPA'] ? 'error' : ''}
          />
          {errors['First Term GPA'] && (
            <span className="error-message">{errors['First Term GPA']}</span>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="second-term-gpa">
            Second Term GPA <span className="required">*</span>
            <HelpIcon text="The student's GPA from their second term. Valid range: 0.0 to 4.5. Comparing first and second term GPAs helps identify trends - declining GPA may indicate risk of not persisting. Students maintaining or improving their GPA are more likely to succeed." />
          </label>
          <input
            id="second-term-gpa"
            type="number"
            step="0.01"
            name="Second Term GPA"
            value={formData['Second Term GPA']}
            onChange={handleChange}
            placeholder="0.0 - 4.0"
            className={errors['Second Term GPA'] ? 'error' : ''}
          />
          {errors['Second Term GPA'] && (
            <span className="error-message">{errors['Second Term GPA']}</span>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="high-school-average">
            High School Average <span className="required">*</span>
            <HelpIcon text="The student's overall average mark from high school. Valid range: 0.0 to 100.0. This reflects their academic preparation before entering post-secondary education. Higher averages indicate stronger foundational knowledge and study habits." />
          </label>
          <input
            id="high-school-average"
            type="number"
            step="0.1"
            name="High School Average"
            value={formData['High School Average']}
            onChange={handleChange}
            placeholder="0 - 100"
            className={errors['High School Average'] ? 'error' : ''}
          />
          {errors['High School Average'] && (
            <span className="error-message">{errors['High School Average']}</span>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="math-score">
            Math Score <span className="required">*</span>
            <HelpIcon text="The student's math score. Valid range: 0.0 to 50.0. This is typically from a standardized test or high school math course. Math skills are often predictive of success in many post-secondary programs, especially in STEM fields." />
          </label>
          <input
            id="math-score"
            type="number"
            step="0.1"
            name="Math Score"
            value={formData['Math Score']}
            onChange={handleChange}
            placeholder="0 - 50"
            className={errors['Math Score'] ? 'error' : ''}
          />
          {errors['Math Score'] && (
            <span className="error-message">{errors['Math Score']}</span>
          )}
        </div>
      </div>
      
      {/* Background Information Section */}
      <div className="form-section">
        <h3>👤 Background Information</h3>
        
        <div className="form-group">
          <label htmlFor="residency">
            Residency
            <HelpIcon text="Student residency status (1=Domestic, 2=International). Domestic students are from the same country as the institution. International students may face additional challenges like language barriers, cultural adjustment, visa issues, or being far from family support networks." />
          </label>
          <select
            id="residency"
            name="Residency"
            value={formData['Residency']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Residency'] &&
              Object.entries(categoricalOptions['Residency']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Residency', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="gender">
            Gender
            <HelpIcon text="The student's gender identity. Options: 1 = Female, 2 = Male, 3 = Neutral. This demographic factor may correlate with different persistence patterns in the historical data. Different genders may face different challenges or have different support systems that can affect academic success." />
          </label>
          <select
            id="gender"
            name="Gender"
            value={formData['Gender']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Gender'] &&
              Object.entries(categoricalOptions['Gender']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Gender', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="funding">
            Funding
            <HelpIcon text="The type of funding or financial aid the student receives. Options: 1 = Apprentice_PS, 2 = GPOG_FT, 3 = Intl Offshore, 4 = Intl Regular, 5 = Intl Transfer, 6 = Joint Program Ryerson, 7 = Joint Program UTSC, 8 = Second Career Program, 9 = Work Safety Insurance Board. Financial support can impact a student's ability to focus on studies. Only codes that appeared in the training data are shown here." />
          </label>
          <select
            id="funding"
            name="Funding"
            value={formData['Funding']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Funding'] &&
              Object.entries(categoricalOptions['Funding']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Funding', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="school">
            School
            <HelpIcon text="The specific school or department within the institution. Possible options: 1 = Advancement, 2 = Business, 3 = Communications, 4 = Community and Health, 5 = Hospitality, 6 = Engineering, 7 = Transportation. IMPORTANT: Only schools that appeared in the training dataset are shown here. If you see only one option (e.g., only '6 - Engineering'), it means the training data only contained students from that school. The model can only make predictions for schools it was trained on. To use other schools, the model would need to be retrained with data from those schools." />
          </label>
          <select
            id="school"
            name="School"
            value={formData['School']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['School'] &&
              Object.entries(categoricalOptions['School']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('School', parseInt(code))}
                </option>
              ))}
          </select>
          {categoricalOptions['School'] && Object.keys(categoricalOptions['School']).length === 1 && (
            <div className="field-warning">
              ⚠️ Only one school available - this is the only school present in the training data.
            </div>
          )}
        </div>
        
        <div className="form-group">
          <label htmlFor="first-language">
            First Language
            <HelpIcon text="The student's first or primary language. Options: 1 = English, 2 = French, 3 = Other. Language barriers can affect academic performance, especially if instruction is in a second language. Students whose first language is not English or French may need additional language support." />
          </label>
          <select
            id="first-language"
            name="First Language"
            value={formData['First Language']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['First Language'] &&
              Object.entries(categoricalOptions['First Language']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('First Language', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="age-group">
            Age Group
            <HelpIcon text="The student's age category. Options: 1 = 0 to 18, 2 = 19 to 20, 3 = 21 to 25, 4 = 26 to 30, 5 = 31 to 35, 6 = 36 to 40, 7 = 41 to 50, 8 = 51 to 60, 9 = 61 to 65, 10 = 66+. Mature students (higher codes) may have different persistence patterns due to life experience, work commitments, and different priorities. Only age groups that appeared in the training data are shown here." />
          </label>
          <select
            id="age-group"
            name="Age Group"
            value={formData['Age Group']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Age Group'] &&
              Object.entries(categoricalOptions['Age Group']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Age Group', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="english-grade">
            English Grade
            <HelpIcon text="The student's English language proficiency level. Options: 1 = Level-130, 2 = Level-131, 3 = Level-140, 4 = Level-141, 5 = Level-150, 6 = Level-151, 7 = Level-160, 8 = Level-161, 9 = Level-170, 10 = Level-171, 11 = Level-180. These represent different English proficiency levels, with higher numbers indicating more advanced skills. Strong English skills are important for academic success. Only levels present in the training data are available." />
          </label>
          <select
            id="english-grade"
            name="English Grade"
            value={formData['English Grade']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['English Grade'] &&
              Object.entries(categoricalOptions['English Grade']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('English Grade', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="prev-education">
            Previous Education
            <HelpIcon text="The student's educational background before entering this program. Options: 1 = HighSchool, 2 = PostSecondary. Students with post-secondary experience may be better prepared for the academic rigor, while high school graduates might need more support transitioning to college/university. Previous education level can indicate preparedness and study skills." />
          </label>
          <select
            id="prev-education"
            name="Prev Education"
            value={formData['Prev Education']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Prev Education'] &&
              Object.entries(categoricalOptions['Prev Education']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Prev Education', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="fast-track">
            Fast Track
            <HelpIcon text="Whether the student is enrolled in an accelerated or fast-track program (1=Yes, 2=No). Fast-track programs compress coursework into shorter timeframes, which may have different persistence rates due to increased intensity and workload. Students in fast-track programs may need additional support." />
          </label>
          <select
            id="fast-track"
            name="Fast Track"
            value={formData['Fast Track']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Fast Track'] &&
              Object.entries(categoricalOptions['Fast Track']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Fast Track', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="coop">
            Co-op Program
            <HelpIcon text="Whether the student is participating in a co-operative education program (1=Yes, 2=No). Co-op programs alternate academic terms with paid work terms, providing real-world experience. This can affect persistence - some students find it motivating, while others may struggle balancing work and study commitments." />
          </label>
          <select
            id="coop"
            name="Coop"
            value={formData['Coop']}
            onChange={handleChange}
          >
            <option value="">Select...</option>
            {categoricalOptions['Coop'] &&
              Object.entries(categoricalOptions['Coop']).map(([code]) => (
                <option key={code} value={code}>
                  {getCategoricalLabel('Coop', parseInt(code))}
                </option>
              ))}
          </select>
        </div>
      </div>
      
      <button type="submit" className="submit-button" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Success'}
      </button>
    </form>
  );
}

export default StudentForm;

