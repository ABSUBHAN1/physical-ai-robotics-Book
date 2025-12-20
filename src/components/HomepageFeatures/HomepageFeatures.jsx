import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Simulation-First Methodology',
    description: (
      <>
        Develop, validate, and iterate in high-fidelity digital twins before physical deployment.
        Our containerized environments guarantee <strong>100% reproducibility</strong> across all research.
      </>
    ),
  },
  {
    title: 'Integrated AI-Robotics Stack',
    description: (
      <>
        Master the complete pipeline from <strong>LLM-based reasoning</strong> to <strong>ROS 2 execution</strong>.
        Bridge cognitive AI with physical motion through modular, production-grade interfaces.
      </>
    ),
  },
  {
    title: 'Academic Research Rigor',
    description: (
      <>
        Each concept traces to peer-reviewed literature. Modules include 
        <strong> required reading</strong>, <strong>citation standards</strong>, and 
        <strong> validation protocols</strong> for graduate-level credibility.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <div className={styles.featureIconContainer}>
          <div className={styles.featureIcon}></div>
        </div>
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Core Pedagogical Approach
        </Heading>
        <p className={styles.sectionSubtitle}>
          Three fundamental principles guiding our curriculum design
        </p>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}