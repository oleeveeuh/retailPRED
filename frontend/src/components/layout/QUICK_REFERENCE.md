# Layout Quick Reference Guide

## Component Structure

```
frontend/src/
├── components/
│   └── layout/
│       ├── Layout.tsx          # Main layout wrapper
│       ├── Sidebar.tsx         # Navigation sidebar
│       ├── Header.tsx          # Top navigation bar
│       ├── Layout.css          # Custom styles
│       └── LAYOUT_REDESIGN.md  # Full documentation
└── index.css                   # Global styles + Tailwind
```

## Quick Start

### Using the Layout

```tsx
import { Layout } from './components/layout';

<Layout>
  <YourPageContent />
</Layout>
```

### Adding Navigation Items

Edit `Sidebar.tsx`:

```tsx
const navItems: NavItem[] = [
  {
    name: 'Page Name',
    path: '/dashboard/page-name',
    icon: IconComponent,
    description: 'Brief description',
  },
];
```

### Using Style Classes

```tsx
// Cards
<div className="glass-card">Content</div>

// Buttons
<button className="btn-primary">Primary</button>
<button className="btn-secondary">Secondary</button>

// Inputs
<input className="input-base" />

// Gradient Text
<h1 className="gradient-text">Heading</h1>

// Hover Effects
<div className="card-hover">Content</div>
```

## Design Tokens

### Colors

```tsx
// Backgrounds
bg-slate-50        // Light
bg-slate-800       // Dark

// Primary
bg-blue-600        // Primary button
hover:bg-blue-700  // Hover state

// Accent
bg-purple-500      // Highlights

// Status
bg-emerald-500     // Success
bg-amber-500       // Warning
bg-red-500         // Error
```

### Spacing

```tsx
p-4  // 16px (sm)
p-6  // 24px (md)
p-8  // 32px (lg)

gap-4  // 16px
gap-6  // 24px
gap-8  // 32px
```

### Shadows

```tsx
shadow-lg       // Large shadow
shadow-xl       // Extra large
shadow-blue-500/50  // Colored shadow (50% opacity)
```

## Animation Classes

```tsx
// Framer Motion
<motion.div
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  transition={{ duration: 0.3 }}
/>

// CSS Animations
className="animate-float"      // Up/down motion
className="animate-shimmer"    // Loading effect
className="animate-pulse"      // Heartbeat
```

## Dark Mode

```tsx
// Enable dark mode
document.documentElement.classList.add('dark');

// Conditional classes
className="bg-white dark:bg-slate-800"
className="text-slate-900 dark:text-slate-100"

// Toggle
const [isDark, setIsDark] = useState(false);
useEffect(() => {
  document.documentElement.classList.toggle('dark', isDark);
}, [isDark]);
```

## Responsive Utilities

```tsx
// Hide on mobile
className="hidden md:block"

// Mobile only
className="block md:hidden"

// Responsive grid
className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3"

// Responsive text
className="text-sm md:text-base lg:text-lg"
```

## Common Patterns

### Page Header

```tsx
<div className="mb-8">
  <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
    Page Title
  </h1>
  <p className="text-slate-600 dark:text-slate-400 mt-2">
    Page description
  </p>
</div>
```

### Card Grid

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  {items.map(item => (
    <div key={item.id} className="glass-card card-hover">
      {item.content}
    </div>
  ))}
</div>
```

### Loading State

```tsx
<div className="space-y-4">
  <div className="skeleton h-4 w-3/4" />
  <div className="skeleton h-4 w-1/2" />
  <div className="skeleton h-32 w-full" />
</div>
```

### Status Indicator

```tsx
<div className="flex items-center space-x-2">
  <div className="status-online" />
  <span className="text-sm">Online</span>
</div>
```

### Action Button

```tsx
<motion.button
  whileHover={{ scale: 1.02 }}
  whileTap={{ scale: 0.98 }}
  className="btn-primary"
>
  Click Me
</motion.button>
```

## Icon Usage

```tsx
import { IconName } from 'lucide-react';

<IconName className="w-5 h-5" />
<IconName className="w-6 h-6" />
<IconName className="w-8 h-8" />
```

## Glassmorphism Effect

```tsx
// Light mode
<div className="
  bg-white/80
  backdrop-blur-xl
  border
  border-slate-200/50
  shadow-xl
">

// Dark mode
<div className="
  bg-slate-800/80
  backdrop-blur-xl
  border
  border-slate-700/50
  shadow-xl
">
```

## Gradient Effects

```tsx
// Text
<span className="gradient-text">Text</span>

// Background
<div className="bg-gradient-to-r from-blue-600 to-purple-600" />

// Border
<div className="gradient-border">
  <div className="bg-white rounded-xl">Content</div>
</div>
```

## Motion Components

### Page Transition

```tsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  exit={{ opacity: 0, y: -20 }}
  transition={{ duration: 0.3 }}
>
  {children}
</motion.div>
```

### Staggered Children

```tsx
{items.map((item, i) => (
  <motion.div
    key={i}
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ delay: i * 0.1 }}
  />
))}
```

### Hover Effect

```tsx
<motion.div
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
>
  Content
</motion.div>
```

## Form Elements

### Input

```tsx
<input
  type="text"
  className="input-base"
  placeholder="Enter text..."
/>
```

### Select

```tsx
<select className="input-base">
  <option>Option 1</option>
  <option>Option 2</option>
</select>
```

### Checkbox

```tsx
<label className="flex items-center space-x-2">
  <input type="checkbox" className="rounded text-blue-600" />
  <span>Label</span>
</label>
```

## Feedback Components

### Success Message

```tsx
<div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-4">
  <div className="flex items-center space-x-3">
    <CheckCircle className="w-5 h-5 text-emerald-600" />
    <p className="text-emerald-800 dark:text-emerald-200">Success message</p>
  </div>
</div>
```

### Error Message

```tsx
<div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
  <div className="flex items-center space-x-3">
    <XCircle className="w-5 h-5 text-red-600" />
    <p className="text-red-800 dark:text-red-200">Error message</p>
  </div>
</div>
```

### Warning Message

```tsx
<div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
  <div className="flex items-center space-x-3">
    <AlertCircle className="w-5 h-5 text-amber-600" />
    <p className="text-amber-800 dark:text-amber-200">Warning message</p>
  </div>
</div>
```

## Accessibility

### Focus Ring

```tsx
<button className="focus-ring">
  Accessible Button
</button>
```

### ARIA Labels

```tsx
<button aria-label="Close dialog">
  <X className="w-6 h-6" />
</button>
```

### Semantic HTML

```tsx
<nav aria-label="Main navigation">
  {/* Navigation */}
</nav>

<main aria-label="Main content">
  {/* Content */}
</main>

<aside aria-label="Supplementary content">
  {/* Sidebar */}
</aside>
```

## Performance Tips

1. **Use CSS animations** over JS when possible
2. **Animate transform/opacity** for 60fps
3. **Lazy load** images and components
4. **Debounce** search inputs
5. **Memoize** expensive computations
6. **Code split** routes

## Browser DevTools

### Check Dark Mode

```js
document.documentElement.classList.toggle('dark');
```

### Check Responsive Design

1. Open DevTools (F12)
2. Click device toolbar (Ctrl+Shift+M)
3. Test at: 375px, 768px, 1024px, 1440px

### Measure Performance

```js
// React DevTools Profiler
// Lighthouse Audit
// Performance tab in DevTools
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Styles not applying | Run `npm run dev` to rebuild Tailwind |
| Dark mode not working | Check localStorage and .dark class on html |
| Animations choppy | Check GPU acceleration, reduce complexity |
| Sidebar not collapsing | Check lg: breakpoint and z-index |
| Layout breaking | Check min-width and overflow settings |

## Resources

- [Framer Motion Docs](https://www.framer.com/motion/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Lucide Icons](https://lucide.dev/)
- [React Router](https://reactrouter.com/)
- [TanStack Query](https://tanstack.com/query/latest)

---

**Need Help?** Check [LAYOUT_REDESIGN.md](./LAYOUT_REDESIGN.md) for detailed documentation.
