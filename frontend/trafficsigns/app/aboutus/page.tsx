import styles from '../styles/AboutUs.module.css'; // Importamos el módulo CSS
import Image from 'next/image';

export default function AboutUs() {
  return (
    <div style={{ fontFamily: 'Trebuchet MS, sans-serif', backgroundColor: 'bg-gray-100' }}>
      <div style={{ color: 'white', fontSize: '30px', width: '100%', backgroundColor: '#0c4160ff', padding: '40px 0', textAlign: 'center' }}>
        <h1>ABOUT US</h1>
      </div>

      <div className={styles.container}>
        {/* Header Section */}
        <div className={styles.footer} style={{ marginTop: '10px'}}>
        <div style={{ marginTop: '10px', textAlign: 'center', maxWidth: '980px', margin: '0 auto', textAlign: "justify" }}>
          <p style={{ marginBottom: '20px' }}>
          We are five students from the Bachelor’s Degree in Data Science and Engineering (GCED) at the Polytechnic University of Catalonia (UPC). Our team is passionate about applying data-driven solutions to real-world problems, and through this project, we aim to deepen our expertise in the fields of Machine Learning and MLOps.
          </p>
          </div>
    </div>

        {/* Team Section */}
        <section className={styles.team}>
          <div className={styles.member}>
            <div className={styles.imageContainer}>
              <img src="/Laia.png" alt="Laia" className={styles.image} />
            </div>
            <p className={styles.name}>Laia Álvarez</p>
          </div>

          <div className={styles.member}>
            <div className={styles.imageContainer}>
              <img src="/Adri.png" alt="Adri" className={styles.image} />
            </div>
            <p className={styles.name}>Adrián Cerezuela</p>
          </div>

          <div className={styles.member}>
            <div className={styles.imageContainer}>
              <img src="/Eva.png" alt="Eva" className={styles.image} />
            </div>
            <p className={styles.name}>Eva Jiménez</p>
          </div>

          <div className={styles.member}>
            <div className={styles.imageContainer}>
              <img src="/Roger.png" alt="Roger" className={styles.image} />
            </div>
            <p className={styles.name}>Roger Martínez</p>
          </div>

          <div className={styles.member}>
            <div className={styles.imageContainer}>
              <img src="/Ramon.png" alt="Ramon" className={styles.image} />
            </div>
            <p className={styles.name}>Ramon Ventura</p>
          </div>
        </section>
        
        {/* Footer Section */}
        <div className={styles.footer} style={{ marginTop: '60px'}}>
        <div style={{ marginTop: '60px', textAlign: 'center', maxWidth: '980px', margin: '0 auto', textAlign: "justify" }}>
          <p style={{ marginBottom: '20px' }}>
          The goal of this project is to develop a Machine Learning model capable of detecting and classifying traffic signs. This is a critical task with wide-reaching applications, especially in areas like autonomous driving and smart city infrastructure. Traffic signs are essential for maintaining road safety, and an automated system for detecting and interpreting these signs can significantly enhance safety by ensuring vehicles and drivers have a better understanding of road regulations.
          </p>
          <p style={{ marginBottom: '20px' }}>
            Beyond addressing this practical issue, the project serves as an opportunity to gain hands-on experience with MLOps tools. We will work with tools for data versioning, code versioning, experiment tracking, and reproducibility to strengthen our understanding of modern Machine Learning operations and best practices.
          </p>
          <p>
            We invite you to explore our project and get to know the team behind it!
          </p>
      </div>
      </div>
      <Image
            src="/Validations.jpg"
            alt="Next.js logo"
            width={500}
            height={500}
            priority
          />
      </div>
    </div>
  );
}
