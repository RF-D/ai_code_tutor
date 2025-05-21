import React, { useState } from 'react';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import Card from '../components/ui/Card';
import { FiSettings, FiArrowRight, FiSearch, FiMail, FiLock, FiCheck, FiAlertCircle, FiCode, FiBox, FiCpu } from 'react-icons/fi';

/**
 * Test page to demonstrate Tailwind CSS and Shadcn/UI functionality
 */
function ShadcnTestPage() {
  const [count, setCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [formValues, setFormValues] = useState({
    username: '',
    email: '',
    password: '',
  });

  const handleClick = () => {
    setCount(count + 1);
  };

  const handleLoadingClick = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: value,
    });
  };

  const handleCardClick = (cardName) => {
    alert(`You clicked on the "${cardName}" card`);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-text-primary">Shadcn/UI and Tailwind CSS Test</h1>
        
        {/* Tailwind Test Section */}
        <div className="rounded-lg border border-border p-6 mb-8 bg-background-secondary">
          <h2 className="text-xl font-semibold mb-4 text-text-primary">Tailwind CSS Test</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="p-4 rounded-md bg-primary/10">
              <p className="font-medium text-primary">Primary Container</p>
              <p className="text-sm text-text-secondary mt-2">
                This box uses Tailwind CSS with our theme variables.
              </p>
            </div>
            
            <div className="p-4 rounded-md bg-secondary/10">
              <p className="font-medium text-secondary">Secondary Container</p>
              <p className="text-sm text-text-secondary mt-2">
                Custom colors defined in tailwind.config.js.
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-md bg-success/10">
              <p className="font-medium text-success">Success</p>
            </div>
            
            <div className="p-4 rounded-md bg-info/10">
              <p className="font-medium text-info">Info</p>
            </div>
            
            <div className="p-4 rounded-md bg-danger/10">
              <p className="font-medium text-danger">Danger</p>
            </div>
          </div>
          
          <div className="mt-6">
            <p className="text-text-tertiary text-sm">
              Responsive elements below will change layout based on screen size.
            </p>
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2">
              <div className="p-3 rounded bg-background-tertiary text-center">Item 1</div>
              <div className="p-3 rounded bg-background-tertiary text-center">Item 2</div>
              <div className="p-3 rounded bg-background-tertiary text-center">Item 3</div>
              <div className="p-3 rounded bg-background-tertiary text-center">Item 4</div>
            </div>
          </div>
        </div>
        
        {/* Shadcn/UI Button Test Section */}
        <div className="rounded-lg border border-border p-6 mb-8 bg-background-secondary">
          <h2 className="text-xl font-semibold mb-4 text-text-primary">Shadcn/UI Button Test</h2>
          
          <div className="space-y-4">
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Button Variants</h3>
              <div className="flex flex-wrap gap-2">
                <Button variant="primary">Primary</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="success">Success</Button>
                <Button variant="danger">Danger</Button>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Button Sizes</h3>
              <div className="flex flex-wrap items-center gap-2">
                <Button variant="primary" size="sm">Small</Button>
                <Button variant="primary" size="md">Medium</Button>
                <Button variant="primary" size="lg">Large</Button>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Button Styles</h3>
              <div className="flex flex-wrap gap-2">
                <Button variant="primary">Default</Button>
                <Button variant="primary" outline>Outline</Button>
                <Button variant="primary" text>Text</Button>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Button States</h3>
              <div className="flex flex-wrap gap-2">
                <Button variant="primary" disabled>Disabled</Button>
                <Button variant="primary" loading={isLoading} onClick={handleLoadingClick}>
                  {isLoading ? 'Loading...' : 'Click to Load'}
                </Button>
                <Button variant="success" onClick={handleClick} iconRight={<FiArrowRight />}>
                  Clicked {count} times
                </Button>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Button with Icons</h3>
              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" iconLeft={<FiSettings />}>Settings</Button>
                <Button variant="secondary" iconRight={<FiArrowRight />}>Next</Button>
                <Button variant="secondary" iconLeft={<FiSettings />} iconRight={<FiArrowRight />}>
                  Configure Next
                </Button>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Full Width Button</h3>
              <Button variant="primary" fullWidth>Full Width Button</Button>
            </div>
          </div>
        </div>
        
        {/* Shadcn/UI Input Test Section */}
        <div className="rounded-lg border border-border p-6 mb-8 bg-background-secondary">
          <h2 className="text-xl font-semibold mb-4 text-text-primary">Shadcn/UI Input Test</h2>
          
          <div className="space-y-4">
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Basic Inputs</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  name="username"
                  label="Username"
                  placeholder="Enter your username"
                  value={formValues.username}
                  onChange={handleInputChange}
                />
                <Input
                  name="email"
                  type="email"
                  label="Email"
                  placeholder="Enter your email"
                  value={formValues.email}
                  onChange={handleInputChange}
                  iconLeft={<FiMail />}
                />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Input Sizes</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Input
                  name="small"
                  placeholder="Small input"
                  size="sm"
                />
                <Input
                  name="medium"
                  placeholder="Medium input"
                  size="md"
                />
                <Input
                  name="large"
                  placeholder="Large input"
                  size="lg"
                />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Input with Icons</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  name="search"
                  placeholder="Search..."
                  iconLeft={<FiSearch />}
                />
                <Input
                  name="password"
                  type="password"
                  placeholder="Enter password"
                  iconLeft={<FiLock />}
                />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Input States</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Input
                  name="required"
                  label="Required Field"
                  placeholder="Required input"
                  required
                  helperText="This field is required"
                />
                <Input
                  name="success"
                  label="Success State"
                  value="Correct value"
                  success
                  iconRight={<FiCheck />}
                  helperText="This looks good!"
                  readOnly
                />
                <Input
                  name="error"
                  label="Error State"
                  value="Invalid input"
                  error
                  iconRight={<FiAlertCircle />}
                  helperText="Please provide a valid value"
                  readOnly
                />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Disabled and Read-only</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  name="disabled"
                  label="Disabled Input"
                  value="Can't edit this"
                  disabled
                  readOnly
                />
                <Input
                  name="readonly"
                  label="Read-only Input"
                  value="Can't edit this either"
                  readOnly
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Shadcn/UI Card Test Section */}
        <div className="rounded-lg border border-border p-6 mb-8 bg-background-secondary">
          <h2 className="text-xl font-semibold mb-4 text-text-primary">Shadcn/UI Card Test</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Basic Cards</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card title="Default Card" subtitle="With title and subtitle">
                  <Card.Body>
                    <p className="text-text-primary">This is a basic card with a header and body.</p>
                  </Card.Body>
                </Card>
                
                <Card>
                  <Card.Body>
                    <h3 className="text-lg font-medium mb-2">No Header</h3>
                    <p className="text-text-primary">This card doesn't have a header section.</p>
                  </Card.Body>
                  <Card.Footer>
                    <div className="flex justify-end">
                      <Button size="sm">Action</Button>
                    </div>
                  </Card.Footer>
                </Card>
                
                <Card title="With Footer">
                  <Card.Body>
                    <p className="text-text-primary">This card has a header, body, and footer.</p>
                  </Card.Body>
                  <Card.Footer>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-text-tertiary">Last updated: Today</span>
                      <Button size="sm" variant="primary">Save</Button>
                    </div>
                  </Card.Footer>
                </Card>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Card Variants</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card variant="default" title="Default">
                  <Card.Body>
                    <p className="text-text-primary">Default card with standard styling.</p>
                  </Card.Body>
                </Card>
                
                <Card variant="flat" title="Flat">
                  <Card.Body>
                    <p className="text-text-primary">Flat card without shadows.</p>
                  </Card.Body>
                </Card>
                
                <Card variant="elevated" title="Elevated">
                  <Card.Body>
                    <p className="text-text-primary">Elevated card with stronger shadow.</p>
                  </Card.Body>
                </Card>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Card Sizes</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card size="sm" title="Small">
                  <Card.Body>
                    <p className="text-text-primary">Small-sized card with less padding.</p>
                  </Card.Body>
                </Card>
                
                <Card size="md" title="Medium">
                  <Card.Body>
                    <p className="text-text-primary">Medium-sized card with default padding.</p>
                  </Card.Body>
                </Card>
                
                <Card size="lg" title="Large">
                  <Card.Body>
                    <p className="text-text-primary">Large-sized card with more padding.</p>
                  </Card.Body>
                </Card>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2 text-text-secondary">Interactive Cards</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card 
                  title="Development" 
                  interactive 
                  onClick={() => handleCardClick('Development')}
                >
                  <Card.Body className="flex items-center">
                    <div className="rounded-full bg-primary/20 p-3 mr-3">
                      <FiCode className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <p className="text-text-primary">Click to view development options</p>
                    </div>
                  </Card.Body>
                </Card>
                
                <Card 
                  title="Products" 
                  interactive 
                  onClick={() => handleCardClick('Products')}
                >
                  <Card.Body className="flex items-center">
                    <div className="rounded-full bg-secondary/20 p-3 mr-3">
                      <FiBox className="w-6 h-6 text-secondary" />
                    </div>
                    <div>
                      <p className="text-text-primary">Click to view product options</p>
                    </div>
                  </Card.Body>
                </Card>
                
                <Card 
                  title="Infrastructure" 
                  interactive 
                  onClick={() => handleCardClick('Infrastructure')}
                >
                  <Card.Body className="flex items-center">
                    <div className="rounded-full bg-success/20 p-3 mr-3">
                      <FiCpu className="w-6 h-6 text-success" />
                    </div>
                    <div>
                      <p className="text-text-primary">Click to view infrastructure options</p>
                    </div>
                  </Card.Body>
                </Card>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ShadcnTestPage;