# Fashion Recommendation System

## Project Description
The Fashion Recommendation System is an innovative application that leverages machine learning to provide personalized fashion recommendations based on user preferences and trends. This project consists of a frontend built with Next.js and a backend powered by a custom API hosted on Render. The system allows users to input their style preferences, receive tailored outfit suggestions..

- **Frontend**: A Next.js application with Tailwind CSS for styling, shadcn/ui components, and next-themes for theme management. The frontend is deployed on Render.
- **Backend**: A  FastAPI deployed on Render.
- **DataBase**: MongoDB

This project demonstrates end-to-end development, including UI/UX design, API integration, and deployment, making it a great example for learning full-stack development with modern tools.

## URLs
- **Live Application**: **https://fashion-recommendation-frontend-1.onrender.com**
- **Deployed API**: **https://fashion-recommendation-rls8.onrender.com/docs**

## Video Demo
Check out the video demo to see the Fashion Recommendation System in action:
- **[Video Demo](https://docs.google.com/document/d/1O-QWtLTrZzV2g-8JHX34mRB_UQFuNlBG5r3sfunm5do/edit?tab=t.0)** 

## Setup Instructions

### Prerequisites
- Node.js (v18.x or later)
- npm (v9.x or later)
- Git
- A code editor (e.g., VS Code)
- An internet connection for dependencies and deployment

### Project Structure
- `fashion-recommendation-frontend/`: Contains the Next.js frontend application.
- `fashion-recommendation-backend/`: Contains the backend API (assumed to be a separate repository or folder; adjust paths accordingly).

### Step-by-Step Setup

#### 1. Clone the Repository
Clone the project to your local machine:
```bash
git clone https://github.com/Joh-Ishimwe/fashion-recommendation-
cd fashion-recommendation-
```
2. Set Up the Backend
Navigate to the backend directory and install dependencies:
```bash
cd fashion-recommendation-backend
npm install
```
-Configure Environment Variables:
Create a .env file in the fashion-recommendation-backend directory.
Add the following:
```bash
MONGO_URI=""
DB_NAME="Fashion-Styles_db"
COLLECTION_NAME="Fashion_data"

```
- Install independences from requirent.txt

## 3. Set Up the Frontend
Navigate to the frontend directory and install dependencies:
```bash
cd ../fashion-recommendation-frontend
npm install
```
## Configure Environment Variables:
Create a .env.local file in the fashion-recommendation-frontend directory.

-Initialize shadcn/ui (if not already done):
```bash
npx shadcn@latest init
```
-Add the theme-provider component manually (as done previously)
```bash
mkdir -p components/ui
```
-Run it Locally:
```bash
npm run dev
```
