
import { Button, Grid, makeStyles } from '@material-ui/core';
import { PDFExport, savePDF } from '@progress/kendo-react-pdf';
import { useRef } from 'react';
import React, { useState } from "react";
import { useSelector } from "react-redux";
import logo from "../emily/Emily Logo-01.png"
import { useHistory, useLocation } from 'react-router';



const useStyles = makeStyles({
    root: {
        flexGrow: 1,
    },
    grids: {
        paddingLeft: '10px'
    },
    style_Grid: {

        minHeight: "100vh",
        width: "683px",
        height: "100%",
        margin: "auto",
        boxShadow: "0px 3px 6px rgba(0, 0, 0, 0.2)",
    },
    rep_text: {
        fontSize: "14px",
        fontWeight: 'bold',
        color: "#3585da",

    },
    rep_title: {
        fontWeight: 'bold',
        color: "#3585da",
        margin: 'auto',


    },
    rep_title2: {
        fontSize: "14px",
        fontWeight: 'bold',
        color: "#1061b0",


    },
})

function ExportExample() {
    const classes = useStyles();

    const pdfExportComponent = useRef(null);
    const contentArea = useRef(null);

    const handleExportWithComponent = (event) => {
        pdfExportComponent.current.save();
    }

    const handleExportWithFunction = (event) => {
        savePDF(contentArea.current,);
    }
    const user = useSelector((state) => state.states.user)

    const [dot, setdot] = React.useState(new Date(Date.now()).toUTCString());
    const [dep, setdep] = React.useState(true);
    const [level, setlevel] = React.useState("MILD");
    const [score, setscore] = React.useState("11");
    // ^ report result object 
    const location = useLocation();
    const history = useHistory();

    const getAge = (dateString) => {
        var today = new Date();
        var birthDate = new Date(dateString);
        var age = today.getFullYear() - birthDate.getFullYear();
        var m = today.getMonth() - birthDate.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
            age--;
        }
        return age;
    }

    React.useEffect(() => {
        if (location.state) {
            console.log(location.state)
            setdep(location.state.score < 10 ? false : location.state.dep)
            setscore(location.state.score)
            if (location.state.score === 0) {
                setlevel('Non Depressed')
            }
            else if (location.state.score > 0 && location.state.score < 5) {
                setlevel('Minimal Depressed')
            }
            else if (location.state.score >= 5 && location.state.score < 10) {
                setlevel('Mild Depression')
            }
            else if (location.state.score >= 9 && location.state.score < 15) {
                setlevel('Moderate Depression')
            }
            else if (location.state.score >= 15 && location.state.score < 20)
                setlevel('Moderate Severe Depression')
            else
                setlevel('Severe Depression')
        }
    }, [])

    return (
        <div className="app-content">
            <PDFExport ref={pdfExportComponent} >
                <Grid className={classes.style_Grid} container >
                    <Button primary={true} onClick={handleExportWithComponent}>Download report</Button>
                    <Button primary={true} onClick={() => history.push("/")}>Home</Button >

                    <Grid container xs="12" direction="row"
                        justifyContent="space-between"
                        alignItems="flex-start">
                        <Grid item xs="8">
                            <img src={logo} alt="logo" style={{ width: '60px', paddingRight: '20px', paddingLeft: '20px' }} />
                        </Grid>
                        <Grid item xs="4">
                            <h5 className={classes.rep_text} >PATIENT ID: {user ? user._id : '12345678'}</h5>
                        </Grid>
                    </Grid>
                    <Grid className={classes.rep_title} xs="12">
                        <h1>SESSION REPORT</h1>
                    </Grid>
                    <Grid container className={classes.grids} xs="12" direction="row"
                        justifyContent="space-between"
                        alignItems="flex-start">
                        <Grid item xs="4">
                            <h5 className={classes.rep_text} >NAME: {user ? user.name : 'ABDUL HADI ABID'} </h5>

                        </Grid>

                        <Grid item xs="3">
                            <h5 className={classes.rep_text} >DOB: {user ? user.dob : '23 May 1999'}</h5>

                        </Grid>
                        <Grid item xs="3">
                            <h5 className={classes.rep_text}>AGE: {user ? getAge(user.dob) : '22'} yrs</h5>

                        </Grid>
                        <Grid item xs="3">
                            <h5 className={classes.rep_text} >GENDER: {user ? user.gender : 'MALE'} </h5>
                        </Grid>
                        <Grid item xs="8">
                            <h5 className={classes.rep_text} >DATE AND TIME OF VISIT: {dot}</h5>
                        </Grid>
                    </Grid>
                    <Grid className={classes.rep_title2} xs="12">
                        <h1>INITIAL ANALYSIS</h1>
                    </Grid>
                    <div style={{
                        width: '65%',
                        textJustify: "inter-word",
                        padding: '10px',
                        color: "#1061b0",
                        margin: 'auto',
                        alignItems: 'baseline'
                    }}>
                        <hr style={{
                            height: '3px',
                            backgroundColor: '#2d324d',
                        }} />
                        <div style={{
                            display: 'flex',
                            flexDirection: 'row',
                            justifyContent: 'space-between'
                        }}>
                            <div style={{
                                width: '35%',
                                textAlign: 'center',
                                margin: 'auto'
                            }}>
                                <h1>ANALYSIS</h1>
                            </div>
                            <hr style={{
                                width: '3px',
                                backgroundColor: '#2d324d'
                            }} />
                            <div style={{
                                width: '60%'
                            }}>
                                <h2>DEPRESSION: {dep ? 'YES' : 'NO'}</h2>
                                <h2>LEVEL: {level}</h2>
                                <h2>PHQ-8 SCORE: {score}</h2>
                            </div>

                        </div>
                    </div>

                    <div style={{

                        padding: '10px',
                        color: "#1061b0",
                    }}>
                        <h1 style={{
                            textDecoration: 'underline'
                        }}>EXERCISES: </h1>
                        <div style={{ paddingLeft: '10px' }}>
                            <p>One vigirous excersise session can help alleviate symptoms for hours and a regular
                                schedule may significantly reduce them over time.</p>
                            <p>One vigirous excersise session can help alleviate symptoms for hours and a regular
                                schedule may significantly reduce them over time</p>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'row-reverse' }}>
                            <p>END OF REPORT </p>
                        </div>
                    </div>
                    <hr style={{
                        height: '3px',
                        backgroundColor: '#2d324d'
                    }} />
                    <div style={{
                        fontFamily: "Corporate Logo Rounded",
                        padding: '10px',
                        color: "#1061b0",
                    }}>
                        <p>NOTE: This report is computer generated and it depicts the condition of your
                            depression as best as a machine could analyze you. Although a machine cannot analyze
                            you 100% correct if certain conditions are not metfor the machine.</p>
                    </div>
                    <Button primary={true} onClick={handleExportWithComponent}>Download report</Button>
                </Grid>
            </PDFExport>
        </div>
    );
}

export default ExportExample;