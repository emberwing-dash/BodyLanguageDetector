// InvestorDashboard.js

import React, { useState } from 'react';
import {
  AppBar, Box, Toolbar, Typography, IconButton, Tabs, Tab, Paper, Grid, Card, CardContent, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Avatar, useTheme
} from '@mui/material';
import {
  Brightness4, Brightness7, BusinessCenter, TrendingUp, AccountBalance, Download, Person
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';

const themeColors = {
  primary: '#7b2cbf',
  secondary: '#9d4edd',
  success: '#00b894',
  warning: '#fdcb6e',
  error: '#d63031',
  backgroundLight: '#f9f9fb',
  backgroundDark: '#121212',
  cardLight: '#ffffff',
  cardDark: '#1e1e1e',
  textLight: '#333333',
  textDark: '#ffffff',
  border: 'rgba(0,0,0,0.1)'
};

const metricData = [
  { icon: <BusinessCenter />, label: 'Total Investments', value: '$1.2M' },
  { icon: <TrendingUp />, label: 'Avg ROI', value: '18.5%' },
  { icon: <AccountBalance />, label: 'Startups Funded', value: '32' },
  { icon: <Person />, label: 'Pending Matches', value: '5' }
];

const chartData = [
  { name: 'Jan', roi: 4 },
  { name: 'Feb', roi: 6 },
  { name: 'Mar', roi: 10 },
  { name: 'Apr', roi: 12 },
  { name: 'May', roi: 8 }
];

const investments = [
  { id: 1, startup: 'GreenHive', amount: '$100K', stage: 'Seed', date: '2024-08-10' },
  { id: 2, startup: 'TechBloom', amount: '$250K', stage: 'Series A', date: '2024-09-12' },
  { id: 3, startup: 'MediSwift', amount: '$75K', stage: 'Pre-Seed', date: '2025-01-20' }
];

export default function InvestorDashboard() {
  const [darkMode, setDarkMode] = useState(false);
  const [tab, setTab] = useState(0);

  const theme = useTheme();
  const handleThemeToggle = () => setDarkMode(!darkMode);
  const handleTabChange = (_, newValue) => setTab(newValue);

  const isDark = darkMode;

  return (
    <Box sx={{ bgcolor: isDark ? themeColors.backgroundDark : themeColors.backgroundLight, minHeight: '100vh', color: isDark ? themeColors.textDark : themeColors.textLight }}>
      <AppBar position="static" sx={{ bgcolor: themeColors.primary, borderRadius: '0 0 12px 12px' }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1, fontWeight: 700 }}>Investor Dashboard</Typography>
          <IconButton onClick={handleThemeToggle} color="inherit">
            {darkMode ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Toolbar>
      </AppBar>

      <Tabs value={tab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto" sx={{
        bgcolor: themeColors.secondary,
        color: '#fff',
        px: 2,
        borderRadius: '12px 12px 0 0'
      }}>
        <Tab label="Overview" />
        <Tab label="Investments" />
        <Tab label="Matches" />
      </Tabs>

      <Box sx={{ p: 3 }}>
        {tab === 0 && (
          <>
            <Grid container spacing={3}>
              {metricData.map((metric, i) => (
                <Grid item xs={12} sm={6} md={3} key={i}>
                  <motion.div whileHover={{ scale: 1.05 }}>
                    <Card sx={{
                      display: 'flex', alignItems: 'center', p: 2,
                      bgcolor: isDark ? themeColors.cardDark : themeColors.cardLight,
                      borderRadius: 3, boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}>
                      <Box sx={{ mr: 2 }}>{metric.icon}</Box>
                      <Box>
                        <Typography variant="body2">{metric.label}</Typography>
                        <Typography variant="h6" sx={{ fontWeight: 700 }}>{metric.value}</Typography>
                      </Box>
                    </Card>
                  </motion.div>
                </Grid>
              ))}
            </Grid>

            <Box mt={4}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>ROI Trends</Typography>
              <Card sx={{
                p: 2, borderRadius: 3,
                bgcolor: isDark ? themeColors.cardDark : themeColors.cardLight
              }}>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="roi" stroke={themeColors.primary} strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Box>
          </>
        )}

        {tab === 1 && (
          <Box mt={2}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>My Investments</Typography>
            <Card sx={{ borderRadius: 3, bgcolor: isDark ? themeColors.cardDark : themeColors.cardLight }}>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Startup</TableCell>
                      <TableCell>Amount</TableCell>
                      <TableCell>Stage</TableCell>
                      <TableCell>Date</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {investments.map((inv) => (
                      <TableRow key={inv.id}>
                        <TableCell>{inv.startup}</TableCell>
                        <TableCell>{inv.amount}</TableCell>
                        <TableCell>{inv.stage}</TableCell>
                        <TableCell>{inv.date}</TableCell>
                        <TableCell align="right">
                          <Button size="small" sx={{ bgcolor: themeColors.primary, color: '#fff', borderRadius: 2, px: 2, '&:hover': { bgcolor: themeColors.secondary } }}>
                            <Download fontSize="small" sx={{ mr: 1 }} />
                            Export
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Card>
          </Box>
        )}

        {tab === 2 && (
          <Box mt={2}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Potential Matches</Typography>
            <Grid container spacing={3}>
              {['AgroMind', 'EduGenius', 'NanoCare'].map((startup, idx) => (
                <Grid item xs={12} md={4} key={idx}>
                  <Card sx={{
                    p: 2,
                    bgcolor: isDark ? themeColors.cardDark : themeColors.cardLight,
                    borderRadius: 3,
                    display: 'flex',
                    alignItems: 'center',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                  }}>
                    <Avatar sx={{ bgcolor: themeColors.primary, mr: 2 }}>{startup.charAt(0)}</Avatar>
                    <Box>
                      <Typography fontWeight={600}>{startup}</Typography>
                      <Button size="small" sx={{ mt: 1, bgcolor: themeColors.primary, color: '#fff', borderRadius: 2, px: 2, '&:hover': { bgcolor: themeColors.secondary } }}>
                        View Details
                      </Button>
                    </Box>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </Box>
    </Box>
  );
}