import { Link } from "react-router-dom";

import "./home.css";

export default function Home() {

    return (

        <div className="home">

            <div className="overlay" />

            <nav>

                <h2>MRI AI System</h2>

                <div>

                    <Link to="/">Home</Link>

                    <Link to="/upload">Upload</Link>

                    <Link to="/result">Result</Link>

                    <Link to="/about">About</Link>
                    <Link to="/history">History</Link>

                </div>

            </nav>

            <section className="hero">

                <div>

                    <h1>

                        Brain Tumor

                        <br />

                        AI Diagnosis

                    </h1>

                    <p>

                        Automatic Segmentation

                        Classification

                        Visualization

                        using Deep Learning

                    </p>

                    <Link to="/upload">

                        <button>

                            Start Diagnosis

                        </button>

                    </Link>

                </div>

                <div>

                    <img src="https://res.cloudinary.com/dv422sfhh/image/upload/v1782708134/brisc2025_train_01148_me_ax_t1_yvozic.jpg" />

                </div>

            </section>

        </div>

    )

}