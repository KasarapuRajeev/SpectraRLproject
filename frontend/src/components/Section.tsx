import { NavLink, Outlet } from 'react-router-dom'
import { ReactNode } from 'react'

export type SubTab = { to: string; label: ReactNode }
/** Wraps a group of related pages under one sidebar item, with a sub-navigation bar. */
export default function Section({ tabs }: { tabs: SubTab[] }) {
  return <div>
    <nav className="subnav" aria-label="Section">
      {tabs.map(t => <NavLink key={t.to} to={t.to} className={({ isActive }) => (isActive ? 'on' : '')}>{t.label}</NavLink>)}
    </nav>
    <Outlet />
  </div>
}
