
import re
import html
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.extract_text import extract_text
from utils.resume_builder import create_resume_docx


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Career Studio",
    page_icon="📄",
    layout="wide"
)

# =========================================================
# DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background: #f5f6ff;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #30236f, #4e39a4);
}

[data-testid="stSidebar"] * {
    color: white;
}

.hero {
    padding: 28px;
    background: linear-gradient(110deg, #30236f, #7650ed);
    color: white;
    border-radius: 20px;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 32px;
    font-weight: 800;
    color: white;
}

.hero p {
    color: #eeeaff;
    line-height: 1.7;
}

.panel {
    background: white;
    border: 1px solid #e6e5f5;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(40, 30, 100, .04);
}

.section-title {
    color: #272052;
    font-size: 21px;
    font-weight: 750;
    margin: 12px 0 16px;
}

.skill {
    display: inline-block;
    padding: 6px 10px;
    margin: 3px;
    border-radius: 20px;
    background: #ece8ff;
    color: #4c32bc;
    font-size: 12px;
    font-weight: 600;
}

.match {
    background: #d9f3e3;
    color: #176638;
}

.missing {
    background: #ffe1e3;
    color: #a3222b;
}

.job-card {
    background: white;
    border: 1px solid #e6e5f5;
    border-radius: 15px;
    padding: 18px;
    margin: 12px 0;
}

.job-title {
    font-size: 17px;
    font-weight: 750;
    color: #282052;
}

.score {
    color: #6041df;
    font-size: 24px;
    font-weight: 800;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 650;
    min-height: 42px;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #5938d5, #7653ed);
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6e5f5;
    border-radius: 13px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL, JOB DATA AND SKILLS
# =========================================================
SKILLS = [
    "python", "java", "c++", "c#", "sql",
    "html", "css", "javascript", "react",
    "machine learning", "deep learning",
    "natural language processing", "nlp",
    "bert", "transformers", "pytorch", "tensorflow",
    "pandas", "numpy", "data analysis", "power bi",
    "aws", "docker", "git", "mongodb", "mysql",
    "recruitment", "employee relations", "human resources",
    "hr", "marketing", "customer service",
    "project management", "payroll",
    "benefits administration", "training", "compliance",
    "sales", "communication", "leadership", "reporting",
    "coaching", "employee engagement",
    "performance management", "onboarding",
    "talent acquisition", "photoshop", "illustrator",
    "email marketing", "content creation"
]


def extract_skills(text):
    text = str(text).lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


def skill_tags(skills, css_class=""):
    if not skills:
        return "No skills detected by the current dictionary."

    return " ".join(
        f'<span class="skill {css_class}">{html.escape(s)}</span>'
        for s in skills
    )


def profile_to_text(profile):
    fields = [
        profile.get("summary", ""),
        profile.get("skills", ""),
        profile.get("experience", ""),
        profile.get("projects", ""),
        profile.get("education", ""),
        profile.get("certifications", ""),
        profile.get("achievements", "")
    ]
    return "\n".join(str(x) for x in fields if x)


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_data
def load_jobs():
    jobs = pd.read_csv("data/processed/cleaned_jobs.csv")
    jobs = jobs.dropna(subset=["Job Title", "job_text"]).copy()
    jobs["job_text"] = jobs["job_text"].astype(str)

    jobs = jobs.drop_duplicates(
        subset=["Job Title", "job_text"]
    ).reset_index(drop=True)

    if "job_description" not in jobs.columns:
        jobs["job_description"] = jobs["job_text"]

    if "skills" not in jobs.columns:
        jobs["skills"] = ""

    return jobs


@st.cache_data(show_spinner="Preparing job embeddings...")
def encode_job_texts(texts):
    model = load_model()
    return model.encode(
        list(texts),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False
    )


# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "profile": {},
    "resume_text": "",
    "tailored_profile": {},
    "match_results": None,
    "match_resume_skills": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown(
        "<h2>📄 Career Studio</h2>"
        "<p style='font-size:12px;color:#e1dcff'>"
        "AI RESUME & JOB PLATFORM</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📝 Resume Builder",
            "✨ AI Resume Tailoring",
            "🎯 Job Matching",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### Powered by")
    st.write("NLP + Sentence-BERT")
    st.caption("Semantic job matching and skill analysis")


# =========================================================
# PAGE HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>AI Resume Screening & Job Matching</h1>
    <p>
        Create a resume, tailor it to a target job, discover relevant
        opportunities, and understand your skill gaps.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="section-title">Your Career Workspace</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Resume",
        "Created" if st.session_state.profile else "Not created"
    )

    c2.metric(
        "Resume text",
        "Available" if st.session_state.resume_text else "Not uploaded"
    )

    c3.metric(
        "Latest matching",
        "Available" if st.session_state.match_results is not None
        else "Not run"
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🚀 Start here")
    st.write("**1. Resume Builder** — enter your real education, skills, projects and experience.")
    st.write("**2. AI Resume Tailoring** — provide a target job description to reorganize relevant information.")
    st.write("**3. Job Matching** — compare your resume against the job dataset or a specific job.")
    st.write("**4. Review Skill Gaps** — see skills detected in both texts and skills mentioned only in the job.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.profile:
        st.success("Your resume-builder details are available in this session.")

    st.info(
        "This first version keeps data in the current Streamlit session. "
        "It does not yet have user accounts or permanent cloud storage."
    )


# =========================================================
# RESUME BUILDER
# =========================================================
elif page == "📝 Resume Builder":
    st.markdown(
        '<div class="section-title">📝 Create Your Resume</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Fill in your actual details. Leave optional sections blank "
        "if you don't have information to include."
    )

    old = st.session_state.profile

    with st.form("resume_builder_form"):
        st.subheader("Personal details")

        c1, c2 = st.columns(2)

        with c1:
            name = st.text_input("Full name", value=old.get("name", ""))
            email = st.text_input("Email", value=old.get("email", ""))
            phone = st.text_input("Phone", value=old.get("phone", ""))

        with c2:
            location = st.text_input("City / Location", value=old.get("location", ""))
            linkedin = st.text_input("LinkedIn URL", value=old.get("linkedin", ""))
            target_role = st.text_input(
                "Target job title", value=old.get("target_role", "")
            )

        st.subheader("Professional summary")

        summary = st.text_area(
            "Summary",
            value=old.get("summary", ""),
            placeholder="Briefly describe your background, strengths and career goal.",
            height=100
        )

        st.subheader("Skills")

        skills = st.text_area(
            "Skills (comma-separated)",
            value=old.get("skills", ""),
            placeholder="Python, SQL, NLP, Machine Learning, Communication"
        )

        c1, c2 = st.columns(2)

        with c1:
            education = st.text_area(
                "Education",
                value=old.get("education", ""),
                placeholder="Degree | College | Year | Relevant coursework",
                height=130
            )

            projects = st.text_area(
                "Projects",
                value=old.get("projects", ""),
                placeholder="Project name, technologies, your contribution and results",
                height=160
            )

        with c2:
            experience = st.text_area(
                "Work / Internship Experience",
                value=old.get("experience", ""),
                placeholder="Role | Organization | Dates\nResponsibilities and real achievements",
                height=160
            )

            certifications = st.text_area(
                "Certifications",
                value=old.get("certifications", ""),
                placeholder="Certification name | Issuer | Year",
                height=80
            )

            achievements = st.text_area(
                "Achievements",
                value=old.get("achievements", ""),
                placeholder="Awards, hackathons, publications or achievements",
                height=80
            )

        submitted = st.form_submit_button(
            "💾 Save Resume Details",
            type="primary",
            use_container_width=True
        )

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
        elif not (education.strip() or skills.strip() or experience.strip()
                  or projects.strip()):
            st.error("Add at least one education, skills, experience or project section.")
        else:
            profile = {
                "name": name,
                "email": email,
                "phone": phone,
                "location": location,
                "linkedin": linkedin,
                "target_role": target_role,
                "summary": summary,
                "skills": skills,
                "education": education,
                "projects": projects,
                "experience": experience,
                "certifications": certifications,
                "achievements": achievements
            }

            st.session_state.profile = profile
            st.session_state.resume_text = profile_to_text(profile)
            st.session_state.tailored_profile = {}
            st.success("Resume details saved!")

    profile = st.session_state.profile

    if profile:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Resume preview")

        st.markdown(f"## {profile.get('name', '')}")

        contact = " | ".join(
            x for x in [
                profile.get("email", ""),
                profile.get("phone", ""),
                profile.get("location", ""),
                profile.get("linkedin", "")
            ] if x
        )
        st.caption(contact)

        for title, key in [
            ("Professional Summary", "summary"),
            ("Skills", "skills"),
            ("Experience", "experience"),
            ("Projects", "projects"),
            ("Education", "education"),
            ("Certifications", "certifications"),
            ("Achievements", "achievements")
        ]:
            value = profile.get(key, "")
            if value:
                st.markdown(f"### {title}")
                st.write(value)

        st.markdown('</div>', unsafe_allow_html=True)

        docx_bytes = create_resume_docx(profile)

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]+", "_", profile.get("name", "resume")
        ).strip("_") or "resume"

        st.download_button(
            "⬇️ Download Professional Resume (.docx)",
            data=docx_bytes,
            file_name=f"{safe_name}_resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            use_container_width=True
        )


# =========================================================
# AI RESUME TAILORING
# =========================================================
elif page == "✨ AI Resume Tailoring":
    st.markdown(
        '<div class="section-title">✨ Tailor Your Resume to a Job</div>',
        unsafe_allow_html=True
    )

    profile = st.session_state.profile

    if not profile and not st.session_state.resume_text:
        st.warning(
            "First create a resume in Resume Builder, or upload a resume "
            "in the Job Matching section."
        )

    base_text = st.session_state.resume_text

    job_description = st.text_area(
        "Paste the target job description",
        height=220,
        placeholder="Paste the job title, responsibilities, qualifications and required skills."
    )

    if st.button("✨ Analyze and Tailor Resume", type="primary"):
        if not base_text.strip():
            st.error("Create a resume or upload one first.")
        elif not job_description.strip():
            st.error("Paste a job description first.")
        else:
            resume_skills = extract_skills(base_text)
            job_skills = extract_skills(job_description)

            matched = sorted(set(resume_skills) & set(job_skills))
            missing = sorted(set(job_skills) - set(resume_skills))

            model = load_model()

            embeddings = model.encode(
                [base_text, job_description],
                normalize_embeddings=True
            )

            similarity = float(
                cosine_similarity(
                    embeddings[0:1],
                    embeddings[1:2]
                )[0][0]
            )

            # Only use existing profile information.
            tailored = dict(profile)

            if tailored:
                existing_summary = tailored.get("summary", "").strip()

                if not existing_summary:
                    tailored["summary"] = (
                        f"{tailored.get('target_role', 'Professional')} "
                        "with experience and skills described in the "
                        "resume below."
                    )

                # Reorder existing skills to prioritize job-relevant ones.
                original_skills = [
                    x.strip()
                    for x in tailored.get("skills", "").split(",")
                    if x.strip()
                ]

                matched_original = [
                    x for x in original_skills
                    if x.lower() in {s.lower() for s in matched}
                ]

                other_original = [
                    x for x in original_skills
                    if x not in matched_original
                ]

                tailored["skills"] = ", ".join(
                    matched_original + other_original
                )

            st.session_state.tailored_profile = tailored

            st.session_state.tailoring_report = {
                "similarity": similarity,
                "matched": matched,
                "missing": missing,
                "job_description": job_description
            }

    report = st.session_state.get("tailoring_report")

    if report:
        st.markdown("### Job relevance analysis")

        st.metric(
            "Semantic similarity",
            f"{report['similarity']:.1%}"
        )

        st.caption(
            "This score indicates semantic similarity. It is not "
            "a hiring probability or a validated qualification score."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Skills detected in both")
            st.markdown(
                skill_tags(report["matched"], "match"),
                unsafe_allow_html=True
            )

        with col2:
            st.markdown("#### 📌 Job skills not detected")
            st.markdown(
                skill_tags(report["missing"], "missing"),
                unsafe_allow_html=True
            )

        st.markdown("### Tailored resume")

        tailored = st.session_state.tailored_profile

        if tailored:
            st.info(
                "The builder preserves your provided facts and "
                "prioritizes relevant skills. Review and edit the "
                "summary and experience before sending the resume."
            )

            st.text_area(
                "Professional summary",
                value=tailored.get("summary", ""),
                key="tailored_summary_edit",
                height=100
            )

            st.text_area(
                "Prioritized skills",
                value=tailored.get("skills", ""),
                key="tailored_skills_edit",
                height=100
            )

            if st.button("Save Tailored Version"):
                updated = dict(tailored)
                updated["summary"] = st.session_state.get(
                    "tailored_summary_edit", ""
                )
                updated["skills"] = st.session_state.get(
                    "tailored_skills_edit", ""
                )

                st.session_state.tailored_profile = updated
                st.session_state.profile = updated
                st.session_state.resume_text = profile_to_text(updated)

                st.success("Tailored resume saved to the current session.")

            download_profile = dict(tailored)
            download_profile["summary"] = st.session_state.get(
                "tailored_summary_edit",
                tailored.get("summary", "")
            )
            download_profile["skills"] = st.session_state.get(
                "tailored_skills_edit",
                tailored.get("skills", "")
            )

            st.download_button(
                "⬇️ Download Tailored Resume",
                data=create_resume_docx(download_profile),
                file_name="tailored_resume.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.warning(
                "Your uploaded resume can be analyzed, but structured "
                "resume rewriting works best with the Resume Builder. "
                "Imported PDF/DOCX text does not automatically identify "
                "all education and experience fields."
            )


# =========================================================
# JOB MATCHING
# =========================================================
elif page == "🎯 Job Matching":
    st.markdown(
        '<div class="section-title">🎯 Resume–Job Matching</div>',
        unsafe_allow_html=True
    )

    upload = st.file_uploader(
        "Upload resume (PDF, DOCX or TXT)",
        type=["pdf", "docx", "txt"],
        key="matching_upload"
    )

    if upload is not None:
        try:
            uploaded_text = extract_text(upload)

            if uploaded_text.strip():
                st.session_state.matching_resume_text = uploaded_text
        except Exception as e:
            st.error(f"Could not read resume: {e}")

    use_created = st.checkbox(
        "Use my resume from Resume Builder",
        value=bool(st.session_state.profile)
    )

    resume_for_matching = ""

    if use_created and st.session_state.profile:
        resume_for_matching = profile_to_text(
            st.session_state.profile
        )
    elif st.session_state.get("matching_resume_text"):
        resume_for_matching = st.session_state.matching_resume_text
    elif st.session_state.resume_text:
        resume_for_matching = st.session_state.resume_text

    job_input_mode = st.radio(
        "Choose matching mode",
        ["Search my job dataset", "Match a specific job description"],
        horizontal=True
    )

    specific_job = ""

    if job_input_mode == "Match a specific job description":
        specific_job = st.text_area(
            "Paste job description",
            height=180
        )

    if st.button("🔎 Run Job Matching", type="primary"):
        if not resume_for_matching.strip():
            st.error("Create or upload a resume first.")
        else:
            model = load_model()
            resume_embedding = model.encode(
                [resume_for_matching],
                normalize_embeddings=True
            )

            if job_input_mode == "Match a specific job description":
                if not specific_job.strip():
                    st.error("Paste a job description.")
                    st.stop()

                job_embedding = model.encode(
                    [specific_job],
                    normalize_embeddings=True
                )

                score = float(
                    cosine_similarity(
                        resume_embedding, job_embedding
                    )[0][0]
                )

                st.metric("Semantic similarity", f"{score:.1%}")

                resume_skills = extract_skills(resume_for_matching)
                job_skills = extract_skills(specific_job)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Skills detected in both")
                    st.markdown(
                        skill_tags(
                            sorted(set(resume_skills) & set(job_skills)),
                            "match"
                        ),
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown("#### Skills not detected in resume")
                    st.markdown(
                        skill_tags(
                            sorted(set(job_skills) - set(resume_skills)),
                            "missing"
                        ),
                        unsafe_allow_html=True
                    )

            else:
                with st.spinner("Ranking available jobs..."):
                    jobs = load_jobs()

                    job_embeddings = encode_job_texts(
                        tuple(jobs["job_text"].tolist())
                    )

                    scores = cosine_similarity(
                        resume_embedding, job_embeddings
                    )[0]

                    jobs["similarity"] = scores

                    results = jobs.sort_values(
                        "similarity", ascending=False
                    ).head(20).reset_index(drop=True)

                    st.session_state.match_results = results
                    st.session_state.match_resume_skills = (
                        extract_skills(resume_for_matching)
                    )

    results = st.session_state.match_results

    if results is not None:
        st.markdown("### Ranked job results")

        for rank, (_, row) in enumerate(
            results.iterrows(), start=1
        ):
            st.markdown('<div class="job-card">', unsafe_allow_html=True)

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f'<div class="job-title">{rank}. '
                    f'{html.escape(str(row["Job Title"]))}</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    str(row.get("job_description", ""))[:500]
                )

            with col2:
                st.markdown(
                    f'<div class="score">'
                    f'{float(row["similarity"]):.1%}</div>',
                    unsafe_allow_html=True
                )

            job_skills = extract_skills(
                str(row.get("job_text", "")) + " "
                + str(row.get("skills", ""))
            )

            resume_skills = st.session_state.match_resume_skills

            matched = sorted(set(resume_skills) & set(job_skills))
            missing = sorted(set(job_skills) - set(resume_skills))

            with st.expander("View skill comparison"):
                st.markdown("**Matched skills**")
                st.markdown(
                    skill_tags(matched, "match"),
                    unsafe_allow_html=True
                )

                st.markdown("**Potentially missing skills**")
                st.markdown(
                    skill_tags(missing, "missing"),
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download matching results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="job_matching_results.csv",
            mime="text/csv"
        )


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "AI Career Studio | NLP + Sentence-BERT | "
    "Results are advisory and should be reviewed by the user."
)

