const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const { send } = require("../config/SeenMessage2");
const client = require("../config/MongoDB");

const router = express.Router();
router.get("/:idPatient", async (req, res) => {
    const idPatient = req.params.idPatient;
    const db = client.db("Patients");
    const collection = db.collection("medicalSystem");
    const patient = await collection.findOne({
        idpatient: idPatient
    });

    if (!patient) {
        return res.status(404).json({
            message: "Không tìm thấy bệnh nhân"
        });
    }
    else {
        return res.status(200).json({
            message: "Tìm thấy bệnh nhân",
            patient: patient
        });
    }

});
module.exports = router;