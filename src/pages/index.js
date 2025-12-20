import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import { useEffect, useRef, useState } from 'react';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const [typedText, setTypedText] = useState('');
  const [cursorVisible, setCursorVisible] = useState(true);
  const fullText = 'From Digital Intelligence to Embodied Machines';
  
  useEffect(() => {
    let i = 0;
    const typingInterval = setInterval(() => {
      if (i <= fullText.length) {
        setTypedText(fullText.slice(0, i));
        i++;
      } else {
        clearInterval(typingInterval);
      }
    }, 50);

    const cursorInterval = setInterval(() => {
      setCursorVisible(prev => !prev);
    }, 500);

    return () => {
      clearInterval(typingInterval);
      clearInterval(cursorInterval);
    };
  }, []);

  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className={styles.heroGridBackground}></div>
      <div className="container">
        <div className={styles.titleContainer}>
          <Heading as="h1" className="hero__title">
            <span className={styles.gradientText}>Physical AI &</span>
            <br />
            <span className={styles.gradientTextReverse}>Humanoid Robotics</span>
          </Heading>
        </div>
        <div className={styles.typingContainer}>
          <p className="hero__subtitle">
            {typedText}
            <span className={cursorVisible ? styles.cursor : styles.cursorHidden}>|</span>
          </p>
        </div>
        <div className={styles.buttons}>
          <Link
            className={`button button--secondary button--lg ${styles.ctaButton}`}
            to="/docs/module-1/architecture-of-embodiment">
            <span className={styles.buttonGlow}></span>
            <span className={styles.buttonText}>
              <code>&gt;</code> Start Building Intelligent Humanoid Robots
            </span>
          </Link>
        </div>
        <div className={styles.scrollIndicator}>
          <div className={styles.scrollMouse}>
            <div className={styles.scrollWheel}></div>
          </div>
          <span className={styles.scrollText}>SCROLL</span>
        </div>
      </div>
    </header>
  );
}

function CorePillars() {
  const pillars = [
    {
      title: 'Research-Backed Accuracy',
      icon: '📚',
      description: 'Every concept grounded in peer-reviewed literature and validated engineering principles. No speculative content.',
    },
    {
      title: 'Full Reproducibility',
      icon: '🔁',
      description: 'Dockerized environments, version-pinned dependencies, and simulation-first workflows guarantee identical results.',
    },
    {
      title: 'Integrated AI-Robotics Stack',
      icon: '⚙️',
      description: 'Principled integration of LLMs/VLMs with ROS 2, MoveIt 2, and Nav2 through modular interfaces.',
    },
    {
      title: 'Real-World Task Proficiency',
      icon: '🎯',
      description: 'Learning measured by concrete task completion metrics in simulation, with clear paths to physical deployment.',
    },
  ];

  return (
    <section className={styles.pillars}>
      <div className="container">
        <h2 className={styles.sectionTitle}>
          <span className={styles.sectionTitleNumber}>01</span>
          <span className={styles.sectionTitleText}>Core Principles</span>
          <div className={styles.titleUnderline}></div>
        </h2>
        <div className="row">
          {pillars.map((pillar, idx) => (
            <div className="col col--3" key={idx}>
              <div className={styles.pillarCard} data-aos="fade-up" data-aos-delay={idx * 100}>
                <div className={styles.pillarIcon}>{pillar.icon}</div>
                <div className={styles.pillarNumber}>{String(idx + 1).padStart(2, '0')}</div>
                <h3>{pillar.title}</h3>
                <p>{pillar.description}</p>
                <div className={styles.pillarHoverLine}></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TechStack() {
  const technologies = [
    { name: 'ROS 2 Humble/Iron', color: '#223344', category: 'Framework' },
    { name: 'Gazebo & Ignition', color: '#556677', category: 'Simulation' },
    { name: 'NVIDIA Isaac Sim', color: '#76b900', category: 'Simulation' },
    { name: 'Unity ROS-TCP', color: '#000000', category: 'Simulation' },
    { name: 'OpenAI GPT API', color: '#10a37f', category: 'AI' },
    { name: 'Whisper Speech', color: '#10a37f', category: 'AI' },
    { name: 'PyTorch', color: '#ee4c2c', category: 'AI' },
    { name: 'Nav2 & MoveIt 2', color: '#223344', category: 'Framework' },
    { name: 'Docker', color: '#2496ed', category: 'DevOps' },
    { name: 'RViz2', color: '#223344', category: 'Visualization' },
  ];

  const [hoveredTech, setHoveredTech] = useState(null);

  return (
    <section className={styles.techStack}>
      <div className="container">
        <h2 className={styles.sectionTitle}>
          <span className={styles.sectionTitleNumber}>02</span>
          <span className={styles.sectionTitleText}>Tech Stack</span>
          <div className={styles.titleUnderline}></div>
        </h2>
        <div className={styles.techGrid}>
          {technologies.map((tech, idx) => (
            <div 
              className={`${styles.techItem} ${hoveredTech === idx ? styles.techItemHovered : ''}`}
              key={idx}
              onMouseEnter={() => setHoveredTech(idx)}
              onMouseLeave={() => setHoveredTech(null)}
              style={{
                '--tech-color': tech.color,
                '--animation-delay': `${idx * 0.1}s`
              }}
              data-aos="zoom-in"
              data-aos-delay={idx * 50}
            >
              <div className={styles.techIcon}>
                <div className={styles.techIconInner}></div>
              </div>
              <div className={styles.techContent}>
                <span className={styles.techName}>{tech.name}</span>
                <span className={styles.techCategory}>{tech.category}</span>
              </div>
              <div className={styles.techHoverEffect}></div>
            </div>
          ))}
        </div>
        <div className={styles.techLegend}>
          <div className={styles.legendItem}>
            <div className={styles.legendColor} style={{backgroundColor: '#223344'}}></div>
            <span>Framework</span>
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendColor} style={{backgroundColor: '#556677'}}></div>
            <span>Simulation</span>
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendColor} style={{backgroundColor: '#10a37f'}}></div>
            <span>AI Models</span>
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendColor} style={{backgroundColor: '#2496ed'}}></div>
            <span>DevOps</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function LearningOutcomes() {
  const outcomes = [
    'Architect and implement Vision-Language-Action (VLA) pipelines for humanoid robots',
    'Develop whole-body controllers for bipedal mobility and dexterous manipulation',
    'Integrate LLMs as high-level task planners with ROS 2 execution frameworks',
    'Construct digital twin simulations for safe training and validation of complex behaviors',
    'Deploy reproducible, containerized research workloads for peer verification',
    'Systematically evaluate performance, safety, and failure boundaries of Physical AI systems',
  ];

  const [activeOutcome, setActiveOutcome] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveOutcome(prev => (prev + 1) % outcomes.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className={styles.outcomes}>
      <div className="container">
        <h2 className={styles.sectionTitle}>
          <span className={styles.sectionTitleNumber}>03</span>
          <span className={styles.sectionTitleText}>Learning Outcomes</span>
          <div className={styles.titleUnderline}></div>
        </h2>
        <div className="row">
          <div className="col col--8 col--offset-2">
            <div className={styles.outcomeVisualization}>
              <div className={styles.outcomeDiagram}>
                {outcomes.map((_, idx) => (
                  <div 
                    key={idx}
                    className={`${styles.diagramNode} ${activeOutcome === idx ? styles.diagramNodeActive : ''}`}
                    onClick={() => setActiveOutcome(idx)}
                  >
                    <div className={styles.nodeDot}></div>
                    <div className={styles.nodeLine}></div>
                  </div>
                ))}
              </div>
              <div className={styles.outcomeContent}>
                <div className={styles.outcomeCounter}>
                  <span className={styles.currentOutcome}>{String(activeOutcome + 1).padStart(2, '0')}</span>
                  <span className={styles.totalOutcomes}>/{String(outcomes.length).padStart(2, '0')}</span>
                </div>
                <ul className={styles.outcomeList}>
                  {outcomes.map((outcome, idx) => (
                    <li 
                      key={idx} 
                      className={activeOutcome === idx ? styles.outcomeActive : ''}
                    >
                      <div className={styles.outcomeCheck}>
                        <div className={styles.checkCircle}>
                          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                            <path d="M10 3L4.5 8.5L2 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        </div>
                      </div>
                      <span className={styles.outcomeText}>{outcome}</span>
                    </li>
                  ))}
                </ul>
                <div className={styles.outcomeNavigation}>
                  <button 
                    className={styles.navButton}
                    onClick={() => setActiveOutcome(prev => prev > 0 ? prev - 1 : outcomes.length - 1)}
                  >
                    ← Prev
                  </button>
                  <button 
                    className={styles.navButton}
                    onClick={() => setActiveOutcome(prev => prev < outcomes.length - 1 ? prev + 1 : 0)}
                  >
                    Next →
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  const [glitchEffect, setGlitchEffect] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setGlitchEffect(true);
      setTimeout(() => setGlitchEffect(false), 100);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className={styles.ctaSection}>
      <div className={styles.ctaBackground}>
        <div className={styles.ctaBackgroundGrid}></div>
        <div className={styles.ctaBackgroundLines}></div>
      </div>
      <div className="container">
        <div className={styles.ctaContent}>
          <h2 className={glitchEffect ? styles.glitchEffect : ''}>
            <span data-text="Begin">Begin</span>{" "}
            <span data-text="Engineering">Engineering</span>{" "}
            <span data-text="the">the</span>{" "}
            <span data-text="Future">Future</span>
          </h2>
          <p className={styles.ctaSubtitle}>
            Join researchers, engineers, and graduate students building the next generation 
            of autonomous humanoid systems. Start with Module 1: The Architecture of Embodiment.
          </p>
          <div className={styles.ctaButtons}>
            <Link
              className={`button button--primary button--lg ${styles.ctaMainButton}`}
              to="/docs/module-1/architecture-of-embodiment">
              <span className={styles.buttonIcon}>
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path d="M10 0L20 10L10 20L8.75 18.75L17.5 10L8.75 1.25L10 0Z" fill="currentColor"/>
                  <path d="M0 10H20V8.75H0V10Z" fill="currentColor"/>
                </svg>
              </span>
              <span className={styles.buttonLabel}>
                <span className={styles.buttonLabelMain}>Start Module 1</span>
                <span className={styles.buttonLabelSub}>Begin Learning</span>
              </span>
            </Link>
            <Link
              className={`button button--outline button--lg ${styles.ctaSecondaryButton}`}
              to="/blog">
              <span className={styles.buttonIcon}>📖</span>
              Read Research Blog
            </Link>
          </div>
          <div className={styles.ctaFooter}>
            <div className={styles.ctaStats}>
              <div className={styles.stat}>
                <span className={styles.statNumber}>04</span>
                <span className={styles.statLabel}>Modules</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>170+</span>
                <span className={styles.statLabel}>Hours</span>
              </div>
              <div className={styles.stat}>
                <span className={styles.statNumber}>100%</span>
                <span className={styles.statLabel}>Hands-on</span>
              </div>
            </div>
            <div className={styles.ctaConsole}>
              <div className={styles.consoleHeader}>
                <div className={styles.consoleButtons}>
                  <div className={styles.consoleButtonRed}></div>
                  <div className={styles.consoleButtonYellow}></div>
                  <div className={styles.consoleButtonGreen}></div>
                </div>
                <span className={styles.consoleTitle}>terminal</span>
              </div>
              <div className={styles.consoleBody}>
                <code>
                  <span className={styles.prompt}>$</span> cd physical-ai-textbook<br/>
                  <span className={styles.prompt}>$</span> docker-compose up<br/>
                  <span className={styles.prompt}>$</span> ros2 launch humanoid_sim simulation.launch.py<br/>
                  <span className={styles.executing}>&gt; Initializing Physical AI Environment...</span>
                </code>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  
  useEffect(() => {
    // Initialize AOS animations
    if (typeof window !== 'undefined') {
      import('aos').then((AOS) => {
        AOS.init({
          duration: 800,
          once: true,
          offset: 100,
        });
      });
    }
  }, []);

  return (
    <Layout
      title={`${siteConfig.title} - Graduate-Level Technical Textbook`}
      description="A rigorous, simulation-first technical textbook on Physical AI and Humanoid Robotics. Master the integrated stack of LLMs, ROS 2, and simulation for building embodied intelligence systems.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <CorePillars />
        <TechStack />
        <LearningOutcomes />
        <CTASection />
      </main>
    </Layout>
  );
}